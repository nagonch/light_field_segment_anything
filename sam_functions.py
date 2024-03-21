from utils import CONFIG
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids
import torch
from torch import nn
from torchvision.transforms.functional import resize
from segment_anything.utils.amg import batch_iterator


def get_sam(return_generator=True):
    sam = sam_model_registry["vit_b"](checkpoint=CONFIG["model-path"])
    sam = sam.to(device="cuda")
    if not return_generator:
        return sam
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


class SimpleSAM(nn.Module):
    def __init__(self, sam, img_size=(128, 128), points_per_batch=64):
        super().__init__()
        self.img_size = img_size
        self.sam = sam
        self.points_per_batch = points_per_batch
        self.sparse_prompt_emb, self.dense_prompt_emb = self.get_prompt_embeddings()

    @torch.no_grad()
    def get_prompt_embeddings(self, n_points_per_side=32):
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
        batch = self.preprocess_batch(batch)
        image_embeddings = self.sam.image_encoder(batch)
        batch_iteration = zip(
            batch_iterator(self.points_per_batch, self.sparse_prompt_emb),
            batch_iterator(self.points_per_batch, self.dense_prompt_emb),
        )
        masks_batches = []
        iou_predictions_batches = []
        low_res_logits_batches = []
        for sparse_emb, dense_emb in batch_iteration:
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb[0],
                dense_prompt_embeddings=dense_emb[0],
                multimask_output=True,
            )
            batch_shape, n_points, c, w, h = low_res_masks.shape
            masks = self.sam.postprocess_masks(
                low_res_masks.reshape(-1, c, w, h),
                input_size=batch.shape[-2:],
                original_size=self.img_size,
            )
            masks = masks.reshape(
                batch_shape, n_points, c, masks.shape[-2], masks.shape[-1]
            )
            if not return_logits:
                masks = masks > self.sam.mask_threshold
            masks_batches.append(masks)
            iou_predictions_batches.append(iou_predictions)
            low_res_logits_batches.append(low_res_masks)
        masks = torch.stack(masks_batches).permute(1, 0, 2, 3, 4, 5)
        iou_predictions = torch.stack(iou_predictions_batches).permute(1, 0, 2, 3)
        low_res_logits = torch.stack(low_res_logits_batches).permute(1, 0, 2, 3, 4, 5)
        masks = masks.reshape(masks.shape[0], -1, masks.shape[-2], masks.shape[-1])
        iou_predictions = iou_predictions.reshape(
            iou_predictions.shape[0],
            -1,
        )
        low_res_logits = low_res_logits.reshape(
            low_res_logits.shape[0],
            -1,
            low_res_logits.shape[-2],
            low_res_logits.shape[-1],
        )
        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_logits,
        }


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
    masks = result["masks"].to(torch.int32)[0].detach().cpu().numpy()
    for mask in masks:
        plt.imshow(mask, cmap="gray")
        plt.show()
