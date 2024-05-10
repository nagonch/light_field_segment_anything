from utils import CONFIG, get_subview_indices
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, hflip
from segment_anything.utils.amg import (
    batch_iterator,
    MaskData,
    calculate_stability_score,
    batched_mask_to_box,
    mask_to_rle_pytorch,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms
from utils import CONFIG
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np


def get_sam():
    sam = sam_model_registry["vit_b"](checkpoint=CONFIG["model-path"])
    sam = sam.to(device="cuda")
    model = SimpleSAM(sam)

    return model


class SimpleSAM(nn.Module):

    def __init__(
        self,
        sam,
    ):
        super().__init__()
        self.sam = sam
        self.pred_iou_thresh = CONFIG["pred-iou-thresh"]
        self.stability_score_offset = CONFIG["stability-score-offset"]
        self.stability_score_thresh = CONFIG["stability-score-thresh"]
        self.points_per_batch = CONFIG["points-per-batch"]
        self.points_per_batch_filtering = CONFIG["points-per-batch-filtering"]
        self.box_nms_thresh = CONFIG["box-nms-thresh"]
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
    def preprocess_batch(self, batch, resize_only=False):
        batch = resize(
            batch.reshape(-1, batch.shape[-2], batch.shape[-1]),
            (1024, 1024),
            antialias=True,
        ).reshape(batch.shape[0], 3, 1024, 1024)
        if not resize_only:
            batch = torch.stack([self.sam.preprocess(x) for x in batch], dim=0)
        return batch

    @torch.no_grad()
    def get_masks_embeddings(self, masks, img_batch):
        from matplotlib import pyplot as plt

        """
        masks: list([w, h])
        img_batch: [b, 3, u, v]
        """
        mask_u, mask_v = masks[0].shape
        img_u, img_v = img_batch.shape[-2:]
        masks = torch.stack(masks, dim=0).cuda().long()
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        mask_x_list = []
        mask_y_list = []
        for mask in masks:
            mask = torch.tensor(mask).cuda().long()
            mask_x, mask_y = torch.where(mask == 1)
            mask_x_list.append(mask_x)
            mask_y_list.append(mask_y)
            x_min, y_min, x_max, y_max = [
                mask_x.min(),
                mask_y.min(),
                mask_x.max(),
                mask_y.max(),
            ]
            x_mins.append(int(x_min * (img_u / mask_u)))
            y_mins.append(int(y_min * (img_v / mask_v)))
            x_maxs.append(int(x_max * (img_u / mask_u)))
            y_maxs.append(int(y_max * (img_v / mask_v)))
        imgs = []
        for x_min, x_max, y_min, y_max in zip(x_mins, x_maxs, y_mins, y_maxs):
            img_patch = img_batch[:, x_min : x_max + 1, y_min : y_max + 1]
            img_patch = self.preprocess_batch(img_patch[None], resize_only=True)
            imgs.append(img_patch)
        imgs = torch.cat(imgs, dim=0)
        batch_iteration = zip(
            batch_iterator(CONFIG["batch-size"], imgs[:2]),
            batch_iterator(CONFIG["batch-size"], mask_x_list[:2]),
            batch_iterator(CONFIG["batch-size"], mask_y_list[:2]),
        )
        mask_embeddings = []
        for batch, mask_x, mask_y in batch_iteration:
            img_patch_embedding = self.sam.image_encoder(batch[0])
            img_patch_embedding = F.interpolate(
                img_patch_embedding,
                size=(mask_u, mask_v),
                mode="bilinear",
            )
            mask_embedding = img_patch_embedding[:, :, mask_x[0][0], mask_y[0][0]].mean(
                axis=-1
            )
            mask_embeddings.append(mask_embedding)
        mask_embeddings = torch.unbind(torch.cat(mask_embeddings, dim=0))
        return mask_embeddings

    @torch.no_grad()
    def forward(
        self,
        input_batch,
        return_logits=True,
    ):
        u, v = (input_batch.shape[-2], input_batch.shape[-1])
        if u < v:
            aspect = v / u
            new_u = CONFIG["mask-mask-side-size"]
            new_v = new_u * aspect
        else:
            aspect = u / v
            new_v = CONFIG["mask-mask-side-size"]
            new_u = new_v * aspect
        batch = self.preprocess_batch(input_batch)
        image_embeddings = self.sam.image_encoder(batch)
        batch_iteration = zip(
            batch_iterator(self.points_per_batch, self.sparse_prompt_emb),
            batch_iterator(self.points_per_batch, self.dense_prompt_emb),
        )
        masks_batches = []
        iou_predictions_batches = []
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
                original_size=(int(new_u), int(new_v)),
            )
            masks = masks.reshape(batch_shape, -1, masks.shape[-2], masks.shape[-1])
            if not return_logits:
                masks = masks > self.sam.mask_threshold
            masks_batches.append(masks)
            iou_predictions_batches.append(iou_predictions.reshape(batch_shape, -1))
        masks = torch.stack(masks_batches).permute(1, 0, 2, 3, 4)
        iou_predictions = torch.stack(iou_predictions_batches).permute(1, 0, 2)
        masks = masks.reshape(masks.shape[0], -1, masks.shape[-2], masks.shape[-1])
        iou_predictions = iou_predictions.reshape(
            iou_predictions.shape[0],
            -1,
        )
        del masks_batches
        del iou_predictions_batches
        del low_res_masks
        del batch_iteration
        result = self.postprocess_masks(masks, iou_predictions, batch)

        return result

    @torch.no_grad()
    def postprocess_masks(self, masks, iou_predictions, img_batches):
        result = []
        u, v = masks.shape[-2:]
        min_mask_area = int(CONFIG["min-mask-area"] * u * v)
        for mask_batch, iou_pred_batch, img_batch in zip(
            masks, iou_predictions, img_batches
        ):
            batch_iteration = zip(
                batch_iterator(self.points_per_batch_filtering, mask_batch),
                batch_iterator(self.points_per_batch_filtering, iou_pred_batch),
            )
            batch_data = MaskData()
            for mask_batched, iou_pred_batched in batch_iteration:
                data = MaskData(
                    masks=mask_batched[0],
                    iou_preds=iou_pred_batched[0],
                )
                del mask_batched
                del iou_pred_batched
                # Filter by predicted IoU
                if self.pred_iou_thresh > 0.0:
                    keep_mask = data["iou_preds"] > self.pred_iou_thresh
                    data.filter(keep_mask)
                data["stability_score"] = calculate_stability_score(
                    data["masks"],
                    self.sam.mask_threshold,
                    self.stability_score_offset,
                )
                if self.stability_score_thresh > 0.0:
                    keep_mask = data["stability_score"] >= self.stability_score_thresh
                    data.filter(keep_mask)
                data["masks"] = data["masks"] > self.sam.mask_threshold
                data["boxes"] = batched_mask_to_box(data["masks"])
                data["rles"] = mask_to_rle_pytorch(data["masks"])
                del data["masks"]
                batch_data.cat(data)
                del data
                del keep_mask
            keep_by_nms = batched_nms(
                batch_data["boxes"].float(),
                batch_data["iou_preds"],
                torch.zeros_like(batch_data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            batch_data.filter(keep_by_nms)
            batch_data.to_numpy()
            if min_mask_area > 0.0:
                batch_data = SamAutomaticMaskGenerator.postprocess_small_regions(
                    batch_data,
                    min_mask_area,
                    self.box_nms_thresh,
                )
            batch_data["masks"] = [rle_to_mask(rle) for rle in batch_data["rles"]]
            result_batch = {
                "masks": batch_data["masks"],
                "iou_predictions": batch_data["iou_preds"],
                "mask_tokens": self.get_masks_embeddings(
                    [torch.tensor(mask).cuda().long() for mask in batch_data["masks"]],
                    img_batch,
                ),
            }
            result.append(result_batch)
            del result_batch
            del batch_data
            del keep_by_nms
        return result

    @torch.no_grad()
    def segment_LF(self, LF):
        os.makedirs("embeddings", exist_ok=True)
        result_masks = []
        max_segment_num = -1
        s, t, u, v, c = LF.shape
        LF = (
            torch.tensor(LF)
            .cuda()
            .reshape(-1, LF.shape[2], LF.shape[3], LF.shape[4])
            .permute(0, 3, 1, 2)
        )
        iterator = batch_iterator(CONFIG["batch-size"], LF)
        for batch_num, batch in tqdm(enumerate(iterator)):
            result = self.forward(batch[0])
            for item_num, item in enumerate(result):
                item = zip(item["masks"], item["mask_tokens"])
                masks = sorted(item, key=(lambda x: x[0].sum()), reverse=True)
                segments = torch.stack([torch.tensor(mask[0]).cuda() for mask in masks])
                embeddings = [torch.tensor(mask[1]).cuda() for mask in masks]
                segments_result = (
                    torch.zeros((segments.shape[-2], segments.shape[-1])).cuda().long()
                )
                embedding_keys = []
                embedding_values = []
                segment_num = 0
                for embedding, segment in zip(embeddings, segments):
                    segments_result[segment] += segment_num + 1
                    embedding_keys.append(segments_result[segment][0].item())
                    embedding_values.append(
                        (
                            embedding,
                            batch_num * CONFIG["batch-size"] + item_num,
                        )
                    )
                    segment_num = segments_result.max() + 1
                segments = segments_result
                segments[segments != 0] += max_segment_num + 1
                embedding_keys = [
                    int(key + max_segment_num + 1) for key in embedding_keys
                ]
                embeddings_map = dict(zip(embedding_keys, embedding_values))
                torch.save(
                    embeddings_map,
                    f"embeddings/{str(batch_num).zfill(4)}_{str(item_num).zfill(4)}.pt",
                )
                del embeddings_map
                max_segment_num = segments.max()
                result_masks.append(segments)
        result_masks = torch.stack(result_masks).reshape(
            s, t, segments_result.shape[-2], segments_result.shape[-1]
        )
        return result_masks

    @torch.no_grad()
    def postprocess_embeddings(self):
        emd_dict = {}
        for item in os.listdir("embeddings"):
            if item.endswith("pt"):
                emd_dict.update(torch.load(f"embeddings/{item}"))
                os.remove(f"embeddings/{item}")
        torch.save(emd_dict, CONFIG["embeddings-filename"])


if __name__ == "__main__":
    pass
