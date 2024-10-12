from sam2_functions import get_auto_mask_predictor, generate_image_masks
from data import HCIOldDataset
import warnings
from utils import visualize_segmentation_mask
import torch
from torchvision.transforms.functional import resize
import numpy as np
import yaml
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

with open("LF_sam_image_seg.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def LF_image_sam_seg(mask_predictor, LF, filename):
    return


if __name__ == "__main__":
    os.makedirs(CONFIG["files-folder"], exist_ok=True)
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = HCIOldDataset()
    for i, (LF, _, _) in enumerate(dataset):
        LF = LF[3:-3, 3:-3]
        segments = LF_image_sam_seg(
            mask_predictor,
            LF,
            filename=str(i).zfill(4),
        )
        visualize_segmentation_mask(segments.cpu().numpy(), LF)
        raise
