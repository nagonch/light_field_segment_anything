from utils import CONFIG
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids
import torch
from torch import nn
from torchvision.transforms.functional import resize
from segment_anything.utils.amg import batch_iterator, MaskData
from utils import CONFIG


def get_sam(return_generator=True):
    sam = sam_model_registry["vit_b"](checkpoint=CONFIG["model-path"])
    sam = sam.to(device="cuda")
    if not return_generator:
        return sam
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


class SimpleSAM(nn.Module):

    def __init__(
        self,
        sam,
        points_per_batch=CONFIG["points-per-batch"],
        points_per_batch_filtering=CONFIG["points-per-batch-filtering"],
        pred_iou_thresh=CONFIG["pred-iou-thresh"],
    ):
        super().__init__()
        self.sam = sam
        self.pred_iou_thresh = pred_iou_thresh
        self.points_per_batch = points_per_batch
        self.points_per_batch_filtering = points_per_batch_filtering
        self.sparse_prompt_emb, self.dense_prompt_emb = self.get_prompt_embeddings()

    @torch.no_grad()
    def get_prompt_embeddings(self, n_points_per_side=CONFIG["points-per-side"]):
        input_points = torch.tensor(
            build_all_layer_point_grids(n_points_per_side, 0, 1)[0]
        ).cuda()
        input_labels = (
            torch.tensor([1 for _ in range(input_points.shape[0])])
            .cuda()[None]
            .permute(1, 0)
        )
        new_w, new_h = (1024, 1024)
        old_w, old_h = (1, 1)
        input_points[..., 0] = input_points[..., 0] * (new_w / old_w)
        input_points[..., 1] = input_points[..., 1] * (new_h / old_h)
        input_points = input_points[None].permute(1, 0, 2)
        sparse_prompt_embedding, dense_prompt_embedding = self.sam.prompt_encoder(
            (input_points, input_labels), None, None
        )
        return sparse_prompt_embedding, dense_prompt_embedding

    @torch.no_grad()
    def preprocess_batch(self, batch):
        batch = resize(
            batch.reshape(-1, batch.shape[-2], batch.shape[-1]),
            (1024, 1024),
            antialias=True,
        ).reshape(batch.shape[0], 3, 1024, 1024)
        batch = torch.stack([self.sam.preprocess(x) for x in batch], dim=0)
        return batch

    @torch.no_grad()
    def forward(
        self,
        batch,
        return_logits=True,
    ):
        img_size = (batch.shape[-2], batch.shape[-1])
        batch = self.preprocess_batch(batch)
        image_embeddings = self.sam.image_encoder(batch)
        batch_iteration = zip(
            batch_iterator(self.points_per_batch, self.sparse_prompt_emb),
            batch_iterator(self.points_per_batch, self.dense_prompt_emb),
        )
        masks_batches = []
        iou_predictions_batches = []
        mask_tokens = []
        for sparse_emb, dense_emb in batch_iteration:
            low_res_masks, iou_predictions, mask_tokens_out = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=torch.repeat_interleave(
                    self.sam.prompt_encoder.get_dense_pe(),
                    image_embeddings.shape[0],
                    dim=0,
                ),
                sparse_prompt_embeddings=sparse_emb[0],
                dense_prompt_embeddings=dense_emb[0],
                multimask_output=True,
                output_tokens=True,
            )
            batch_shape, n_points, c, w, h = low_res_masks.shape
            masks = self.sam.postprocess_masks(
                low_res_masks.reshape(-1, c, w, h),
                input_size=batch.shape[-2:],
                original_size=img_size,
            )
            masks = masks.reshape(batch_shape, -1, masks.shape[-2], masks.shape[-1])
            if not return_logits:
                masks = masks > self.sam.mask_threshold
            masks_batches.append(masks)
            iou_predictions_batches.append(iou_predictions.reshape(batch_shape, -1))
            batch_shape, _, _, emb_shape = mask_tokens_out.shape
            mask_tokens.append(mask_tokens_out.reshape(batch_shape, -1, emb_shape))
        masks = torch.stack(masks_batches).permute(1, 0, 2, 3, 4)
        iou_predictions = torch.stack(iou_predictions_batches).permute(1, 0, 2)
        masks = masks.reshape(masks.shape[0], -1, masks.shape[-2], masks.shape[-1])
        iou_predictions = iou_predictions.reshape(
            iou_predictions.shape[0],
            -1,
        )
        mask_tokens = torch.stack(mask_tokens).permute(1, 0, 2, 3)
        mask_tokens = mask_tokens.reshape(batch_shape, -1, emb_shape)
        del masks_batches
        del iou_predictions_batches
        del low_res_masks
        del mask_tokens_out
        del batch_iteration
        result = self.postprocess_masks(masks, iou_predictions, mask_tokens)

        return result

    @torch.no_grad()
    def postprocess_masks(self, masks, iou_predictions, mask_tokens):
        result = []
        for mask_batch, iou_pred_batch, mask_token_batch in zip(
            masks, iou_predictions, mask_tokens
        ):
            batch_iteration = zip(
                batch_iterator(self.points_per_batch_filtering, mask_batch),
                batch_iterator(self.points_per_batch_filtering, iou_pred_batch),
                batch_iterator(self.points_per_batch_filtering, mask_token_batch),
            )
            result_batch = {"masks": [], "iou_predictions": [], "mask_tokens": []}
            for mask_batched, iou_pred_batched, mask_token_batched in batch_iteration:
                data = MaskData(
                    masks=mask_batched[0],
                    iou_preds=iou_pred_batched[0],
                    mask_token_batch=mask_token_batched[0],
                )
                del mask_batched
                del iou_pred_batched
                del mask_token_batched

                # Filter by predicted IoU
                if self.pred_iou_thresh > 0.0:
                    keep_mask = data["iou_preds"] > self.pred_iou_thresh
                    data.filter(keep_mask)
                result_batch["masks"].extend(data["masks"])
                result_batch["iou_predictions"].extend(data["iou_preds"])
                result_batch["mask_tokens"].extend(data["mask_token_batch"])
                del data
            result.append(result_batch)
            del result_batch
        return result


if __name__ == "__main__":
    import os
    from PIL import Image
    import numpy as np
    from segment_anything import SamPredictor
    from torchvision.transforms.functional import resize
    from matplotlib import pyplot as plt
    import imgviz
    from utils import get_LF

    sam = get_sam(return_generator=False)
    simple_sam = SimpleSAM(sam)
    dir = "/home/cedaradmin/blender/lightfield/LFPlane/f00051/png"
    LF = get_LF(dir)
    batch = (torch.as_tensor(LF[0:2, 0]).permute(0, -1, 1, 2).float()).cuda()[:1]
    result = simple_sam(batch)
    print(result)
    raise
    for mask in masks[0]:
        plt.imshow(mask, cmap="gray")
        plt.show()
