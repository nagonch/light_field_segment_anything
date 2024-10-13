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


def get_segment_disparities(masks_central, disparities):
    mask_disparities = torch.zeros((masks_central.shape[0],)).cuda()
    for i, mask_i in enumerate(masks_central):
        mask_disparities[i] = disparities[mask_i].mean().item()
    return mask_disparities


def predict_mask_subview_position(mask, disparity, s, t):
    st = F.normalize(torch.tensor([s, t]).cuda()[None].float())[0]
    uv_0 = torch.nonzero(mask)
    uv = (uv_0 + disparity * st).long()
    u = uv[:, 0]
    v = uv[:, 1]
    uv = uv[(u >= 0) & (v >= 0) & (u <= mask.shape[0]) & (v <= mask.shape[1])]
    mask_result = torch.zeros_like(mask)
    mask_result[uv[:, 0], uv[:, 1]] = 1
    return mask_result


def get_prompt_mask_positions(LF, masks_central, mask_disparities):
    s_size, t_size, u_size, v_size = LF.shape[:4]
    result = (
        torch.zeros((masks_central.shape[0], s_size, t_size, u_size, v_size))
        .cuda()
        .bool()
    )
    for i, (mask, disparity) in enumerate(zip(masks_central, mask_disparities)):
        for s in range(s_size):
            for t in range(t_size):
                result[i][s][t] = predict_mask_subview_position(
                    mask, disparity, s - s_size // 2, t - t_size // 2
                )
    return result


def LF_image_sam_seg(mask_predictor, LF, filename):
    s_central, t_central = LF.shape[0] // 2, LF.shape[1] // 2
    masks_central = generate_image_masks(mask_predictor, LF[s_central, t_central])
    disparities = torch.tensor(get_LF_disparities(LF)).cuda()
    # torch.save(masks_central, "masks_central.pt")
    # torch.save(disparities, "disparities.pt")
    # masks_central = torch.load("masks_central.pt")
    # disparities = torch.load("disparities.pt")
    mask_disparities = get_segment_disparities(masks_central, disparities)
    prompt_mask_positions = get_prompt_mask_positions(
        LF, masks_central, mask_disparities
    )
    s, t, u, v = LF.shape[:4]
    for mask in prompt_mask_positions:
        visualize_segmentation_mask(mask.long().cpu().numpy(), LF)
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
