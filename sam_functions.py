from utils import CONFIG
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids
import torch
from torch import nn
from torchvision.transforms.functional import resize
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
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb[0],
                dense_prompt_embeddings=dense_emb[0],
                multimask_output=True,
                output_tokens=True,
                token_size=CONFIG["token-size"],
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
            batch_data = MaskData()
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
            batch_data["masks"] = [rle_to_mask(rle) for rle in batch_data["rles"]]
            result_batch = {
                "masks": batch_data["masks"],
                "iou_predictions": batch_data["iou_preds"],
                "mask_tokens": batch_data["mask_token_batch"],
            }
            result.append(result_batch)
            del result_batch
            del batch_data
            del keep_by_nms
        return result

    @torch.no_grad()
    def segment_LF(self, LF):
        result_masks = []
        result_embeddings = []
        max_segment_num = -1
        s, t, u, v, c = LF.shape
        min_mask_area = int(CONFIG["min-mask-area"] * u * v)
        LF = (
            torch.tensor(LF)
            .cuda()
            .reshape(-1, LF.shape[2], LF.shape[3], LF.shape[4])
            .permute(0, 3, 1, 2)
        )
        iterator = batch_iterator(CONFIG["subviews-batch-size"], LF)
        for batch in tqdm(iterator):
            result = self.forward(batch[0])
            for item in result:
                item = zip(item["masks"], item["mask_tokens"])
                masks = sorted(item, key=(lambda x: x[0].sum()), reverse=True)
                masks = list(filter(lambda mask: mask[0].sum() >= min_mask_area, masks))
                segments = torch.stack([torch.tensor(mask[0]).cuda() for mask in masks])
                embeddings = [torch.tensor(mask[1]).cuda() for mask in masks]
                segments_result = torch.zeros((u, v)).cuda().long()
                emb_size = embeddings[0].shape[-1]
                embeddings_map = torch.zeros(
                    (
                        u,
                        v,
                        emb_size,
                    )
                ).cuda()
                segment_num = 0
                emb_num = 0
                for segment in segments:
                    segments_result[segment] += segment_num + 1
                    embeddings_map[segment] += embeddings[emb_num]
                    segment_num += 1
                    emb_num += 1
                segments = segments_result
                segments[segments != 0] += max_segment_num + 1
                result_embeddings.append(embeddings_map)
                max_segment_num = segments.max()
                result_masks.append(segments)
        result_masks = torch.stack(result_masks).reshape(s, t, u, v)
        result_embeddings = torch.stack(result_embeddings).reshape(s, t, u, v, emb_size)
        return result_masks, result_embeddings


if __name__ == "__main__":
    pass
