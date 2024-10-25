from sam2_functions import (
    get_auto_mask_predictor,
    generate_image_masks,
    get_video_predictor,
)
from data2 import UrbanLFSynDataset
import warnings
from utils import (
    visualize_segmentation_mask,
    masks_iou,
    save_LF_lawnmower,
    lawnmower_indices,
)
import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from plenpy.lightfields import LightField


def sam2_video_LF_segmentation(LF):
    pass


if __name__ == "__main__":
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = UrbanLFSynDataset(
        "/home/nagonch/repos/LF_object_tracking/UrbanLF_Syn/val"
    )
    for i, (LF, _, _) in enumerate(dataset):
        LightField(LF).show()
