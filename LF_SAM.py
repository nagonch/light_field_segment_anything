import torch
from torch import nn
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import SAM_CONFIG
from data import HCIOldDataset
from torchvision.transforms.functional import resize


class SimpleSAM(nn.Module):

    def __init__(
        self,
        sam_generator,
    ):
        super().__init__()
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
    def forward(
        self,
        input_LF,
    ):
        embedding = self.generator.predictor.model.image_encoder(
            self.preprocess_batch(torch.tensor(input_LF).cuda().permute(2, 0, 1)[None])
        )
        masks = self.generator.generate(input_LF)
        return masks


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
