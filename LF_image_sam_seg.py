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
from plenpy.lightfields import LightField

warnings.filterwarnings("ignore")

with open("LF_sam_image_seg.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def get_LF_disparities(LF):
    LF = LightField(LF)
    disp, _ = LF.get_disparity()
    return np.nan_to_num(disp)


def reduce_masks(masks):
    """
    Convert [N, U, V] masks to [U, V] segments
    The bigger the segment, the smaller the ID
    TODO: move to utils
    """
    areas = masks.sum(dim=(1, 2))
    masks_result = torch.zeros_like(masks[0]).long().cuda()
    for i, mask_i in enumerate(torch.argsort(areas, descending=True)):
        masks_result[masks[mask_i]] = i  # smaller segments on top of bigger ones
    return masks_result


def get_segment_disparities(segments_central, disparities):
    segment_disparities = {}
    for segment_i in torch.unique(segments_central)[1:]:  # exclude 0 (no segment)
        segment_disparities[segment_i.item()] = (
            disparities[segments_central == segment_i].mean().item()
        )
    return segment_disparities


def LF_image_sam_seg(mask_predictor, LF, filename):
    s_central, t_central = LF.shape[0] // 2, LF.shape[1] // 2
    masks_central = generate_image_masks(mask_predictor, LF[s_central, t_central])
    segments_central = reduce_masks(masks_central)
    disparities = torch.tensor(get_LF_disparities(LF)).cuda()
    torch.save(segments_central, "segments.pt")
    torch.save(disparities, "disparity.pt")
    # segments_central = torch.load("segments.pt")
    # disparities = torch.load("disparity.pt")
    print(get_segment_disparities(segments_central, disparities))
    raise
    return


if __name__ == "__main__":
    os.makedirs(CONFIG["files-folder"], exist_ok=True)
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = HCIOldDataset()
    for i, (LF, _, _) in enumerate(dataset):
        LF = LF[3:-3, 3:-3]
        LF_image_sam_seg(
            mask_predictor,
            LF,
            filename=str(i).zfill(4),
        )
