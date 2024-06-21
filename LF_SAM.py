import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import SAM_CONFIG
from data import HCIOldDataset
from torchvision.transforms.functional import resize
from torch.nn import functional as F
import os
from tqdm import tqdm


class SimpleSAM:

    def __init__(
        self,
        sam_generator,
    ):
        self.generator = sam_generator

    @torch.no_grad()
    def preprocess_batch(self, batch, resize_only=False):
        batch = resize(
            batch.reshape(-1, batch.shape[-2], batch.shape[-1]),
            (1024, 1024),
            antialias=True,
        ).reshape(batch.shape[0], 3, 1024, 1024)
        if not resize_only:
            batch = torch.stack(
                [self.generator.predictor.model.preprocess(x) for x in batch], dim=0
            )
        return batch

    @torch.no_grad()
    def get_masks_embeddings(self, masks, img_embedding):
        result_embeddings = []
        for mask in masks:
            mask_x, mask_y = torch.where(mask == 1)
            embeddings = img_embedding[:, :, mask_x, mask_y].mean(axis=-1)
            result_embeddings.append(embeddings)
        result = torch.cat(result_embeddings, dim=0)
        return result

    @torch.no_grad()
    def forward(
        self,
        input_img,
    ):
        u, v, _ = input_img.shape
        img_embedding = self.generator.predictor.model.image_encoder(
            self.preprocess_batch(torch.tensor(input_img).cuda().permute(2, 0, 1)[None])
        )
        img_embedding = F.interpolate(
            img_embedding,
            size=(int(u), int(v)),
            mode="bilinear",
        )
        masks = self.generator.generate(input_img)
        embeddings = self.get_masks_embeddings(masks, img_embedding)
        return masks, embeddings

    @torch.no_grad()
    def segment_LF(self, LF):
        os.makedirs("embeddings", exist_ok=True)
        os.makedirs("segments", exist_ok=True)
        max_segment_num = -1
        self.s, self.t, self.u, self.v, c = LF.shape
        LF = LF.reshape(-1, LF.shape[2], LF.shape[3], LF.shape[4])
        for subview_num, subview in tqdm(enumerate(LF)):
            masks, embeddings = self.forward(subview)
            item = zip(masks, embeddings)
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
                segments_result[segment] = segment_num + 1
                embedding_keys.append(segment_num + 1)
                segment_num += 1
                embedding_values.append(
                    (
                        embedding,
                        subview_num,
                    )
                )
            segments = segments_result
            segments[segments != 0] += max_segment_num
            embedding_keys = [int(key + max_segment_num) for key in embedding_keys]
            embeddings_map = dict(zip(embedding_keys, embedding_values))
            torch.save(
                embeddings_map,
                f"embeddings/{str(subview_num).zfill(4)}.pt",
            )
            torch.save(
                segments,
                f"segments/{str(subview_num).zfill(4)}.pt",
            )
            max_segment_num = segments.max()
            del embeddings_map
            del embeddings
            del embedding_values
            del masks
            del segments_result
            del item
            del segments

    @torch.no_grad()
    def postprocess_data(self):
        emd_dict = {}
        for item in os.listdir("embeddings"):
            if item.endswith("pt"):
                emd_dict.update(torch.load(f"embeddings/{item}"))
                os.remove(f"embeddings/{item}")
        torch.save(emd_dict, SAM_CONFIG["embeddings-filename"])
        del emd_dict
        result_masks = []
        for item in sorted(os.listdir("segments")):
            if item.endswith("pt"):
                result_masks.append(torch.load(f"segments/{item}"))
                os.remove(f"segments/{item}")
        torch.save(
            torch.stack(result_masks).reshape(self.s, self.t, self.u, self.v),
            SAM_CONFIG["segments-filename"],
        )


def get_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CONFIG["model-path"])
    sam = sam.to(device="cuda")
    model = SimpleSAM(
        SamAutomaticMaskGenerator(
            sam,
            points_per_side=SAM_CONFIG["points-per-side"],
            points_per_batch=SAM_CONFIG["points-per-batch"],
            pred_iou_thresh=SAM_CONFIG["pred-iou-thresh"],
            stability_score_thresh=SAM_CONFIG["stability-score-thresh"],
            stability_score_offset=SAM_CONFIG["stability-score-offset"],
            box_nms_thresh=SAM_CONFIG["box-nms-thresh"],
            min_mask_region_area=SAM_CONFIG["min-mask-area"],
        )
    )

    return model


if __name__ == "__main__":
    simple_sam = get_sam()
    HCI_dataset = HCIOldDataset()
    LF, depth, labels = HCI_dataset.get_scene("horses")
    simple_sam.segment_LF(LF)
