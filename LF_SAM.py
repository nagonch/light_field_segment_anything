import torch
from torch import nn
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import SAM_CONFIG
from data import HCIOldDataset
from torchvision.transforms.functional import resize
import numpy as np
from torch.nn import functional as F


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


def get_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CONFIG["model-path"])
    sam = sam.to(device="cuda")
    model = SimpleSAM(SamAutomaticMaskGenerator(sam))

    return model


if __name__ == "__main__":
    simple_sam = get_sam()
    HCI_dataset = HCIOldDataset()
    LF, depth, labels = HCI_dataset.get_scene("horses")
    simple_sam.forward(LF[0, 0])
