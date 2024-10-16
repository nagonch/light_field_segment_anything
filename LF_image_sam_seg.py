from sam2_functions import get_auto_mask_predictor, generate_image_masks
from data import HCIOldDataset, UrbanLFDataset
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


def masks_to_segments(masks):
    """
    Convert [n, s, t, u, v] masks to [s, t, u v] segments
    The bigger the segment, the smaller the ID
    TODO: move to utils
    masks: torch.tensor [n, s, t, u, v] (torch.bool)
    returns: torch.tensor [s, t, u, v] (torch.long)
    """
    s, t, u, v = masks.shape[1:]
    areas = masks[:, s // 2, t // 2].cpu().sum(dim=(1, 2))
    masks_result = torch.zeros((s, t, u, v), dtype=torch.long).cuda()
    for i, mask_i in enumerate(torch.argsort(areas, descending=True)):
        masks_result[masks[mask_i]] = i  # smaller segments on top of bigger ones
    return masks_result


def get_LF_disparities(LF):
    """
    Get disparities for subview [s//2, t//2]
    LF: np.array [s, t, u, v, 3] (np.uint8)
    returns: np.array [u, v] (np.float32)
    """
    LF = LightField(LF)
    disp, _ = LF.get_disparity()
    return np.nan_to_num(disp)


def get_mask_disparities(masks_central, disparities):
    """
    Get mean disparity of each mask
    masks_central: torch.tensor [n, u, v] (torch.bool)
    disparities: np.array [u, v] (np.float32)
    returns: torch.tensor [n] (torch.float32)
    """
    mask_disparities = torch.zeros((masks_central.shape[0],)).cuda()
    for i, mask_i in enumerate(masks_central):
        mask_disparities[i] = disparities[mask_i].mean().item()
    return mask_disparities


def predict_mask_subview_position(mask, disparity, s, t):
    """
    Use mask's disparity to predict its position in (s, t)
    mask: torch.tensor [u, v] (torch.bool)
    disparity: float
    s, t: float
    returns: torch.tensor [u, v] (torch.bool)
    """
    # mask = mask.cpu()
    # disparity = disparity.cpu()
    st = F.normalize(torch.tensor([s, t])[None].float())[0].cuda()
    uv_0 = torch.nonzero(mask)
    uv = (uv_0 - disparity * st).long()
    u = uv[:, 0]
    v = uv[:, 1]
    uv = uv[(u >= 0) & (v >= 0) & (u < mask.shape[0]) & (v < mask.shape[1])]
    mask_result = torch.zeros_like(mask)
    mask_result[uv[:, 0], uv[:, 1]] = 1
    return mask_result


def get_coarse_matching(LF, masks_central, mask_disparities):
    """
    Predict subview masks using disparities
    LF: np.array [s, t, u, v, 3] (np.uint8)
    masks_central: torch.tensor [u, v] (torch.bool)
    mask_disparities: torch.tensor [n] (torch.float32)
    returns: torch.tensor [n, s, t, u, v] (torch.bool)
    """
    s_size, t_size, u_size, v_size = LF.shape[:4]
    result = torch.zeros(
        (masks_central.shape[0], s_size, t_size, u_size, v_size), dtype=torch.bool
    ).cuda()
    for i, (mask, disparity) in enumerate(zip(masks_central, mask_disparities)):
        for s in range(s_size):
            for t in range(t_size):
                result[i][s][t] = predict_mask_subview_position(
                    mask, disparity, s - s_size // 2, t - t_size // 2
                )
    return result


def get_prompts_for_masks(coarse_masks):
    """
    Calculate prompts from coarse masks
    coarse_masks: torch.tensor [n, s, t, u, v] (torch.bool)
    returns: torch.tensor [n, s, t, 2] (torch.float),
             torch.tensor [n, s, t, 4] (torch.float)
    """
    n, s_size, t_size = coarse_masks.shape[:3]
    point_prompts = torch.zeros((n, s_size, t_size, 2), dtype=torch.float).cuda()
    box_prompts = torch.zeros((n, s_size, t_size, 4), dtype=torch.float).cuda()
    for s in range(s_size):
        for t in range(t_size):
            if s == s_size // 2 and t == t_size // 2:
                continue
            for mask_i, mask in enumerate(coarse_masks[s, t]):
                point_prompts_i = torch.nonzero(mask).flip(1)
                box_pormpts_i = torch.tensor(
                    [
                        point_prompts_i[:, 0].min(),
                        point_prompts_i[:, 1].min(),
                        point_prompts_i[:, 0].max(),
                        point_prompts_i[:, 1].max(),
                    ]
                ).cuda()
                point_prompts_i = point_prompts_i[point_prompts_i.shape[0] // 2, :]
                point_prompts[mask_i, s, t] = point_prompts_i
                box_prompts[mask_i, s, t] = box_pormpts_i
    return point_prompts, box_prompts


def get_fine_matching(LF, image_predictor, coarse_masks):
    """
    Predict subview masks using disparities
    LF: np.array [s, t, u, v, 3] (np.uint8)
    image_predictor: SAM2ImagePredictor
    coarse_masks: torch.tensor [n, s, t, u, v] (torch.bool)
    returns: torch.tensor [n, s, t, u, v] (torch.bool)
    """
    # fine_masks = torch.zeros_like(coarse_masks)
    s_size, t_size = LF.shape[:2]
    for s in range(s_size):
        for t in range(t_size):
            if s == s_size // 2 and t == t_size // 2:
                continue
            coarse_segments_st = torch.clone(coarse_masks[:, s, t])
            coarse_masks[:, s, t, :, :] = False
            image_predictor.set_image(LF[s, t])
            for segment_i, segment in enumerate(coarse_segments_st):
                points = torch.nonzero(segment).flip(1)
                box = torch.tensor(
                    [
                        points[:, 0].min(),
                        points[:, 1].min(),
                        points[:, 0].max(),
                        points[:, 1].max(),
                    ]
                ).cuda()
                points = points[points.shape[0] // 2, :][None]
                labels = torch.ones(points.shape[0])
                fine_segment_result, iou_preds, _ = image_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=True,
                )
                fine_segment_result = torch.tensor(fine_segment_result).cuda()
                coarse_masks[segment_i, s, t] = fine_segment_result[
                    np.argmax(iou_preds)
                ]  # replacing coarse masks with fine ones
    return coarse_masks


def LF_image_sam_seg(mask_predictor, LF):
    s_central, t_central = LF.shape[0] // 2, LF.shape[1] // 2
    print("generate_image_masks...", end="")
    masks_central = generate_image_masks(mask_predictor, LF[s_central, t_central])
    print(f"done, shape: {masks_central.shape}")
    disparities = torch.tensor(get_LF_disparities(LF)).cuda()
    print("get_mask_disparities...", end="")
    mask_disparities = get_mask_disparities(masks_central, disparities)
    print(f"done, shape: {mask_disparities.shape}")
    del disparities

    print("get_coarse_matching...", end="")
    coarse_matched_masks = get_coarse_matching(LF, masks_central, mask_disparities)
    print(f"done, shape: {coarse_matched_masks.shape}")
    del mask_disparities
    del masks_central

    image_predictor = mask_predictor.predictor
    # torch.save(coarse_matched_masks, "coarse_matched_masks.pt")
    # coarse_matched_masks = torch.load("coarse_matched_masks.pt")
    point_prompts, box_prompts = get_prompts_for_masks(coarse_matched_masks)
    print("get_fine_matching...", end="")
    fine_matched_masks = get_fine_matching(LF, image_predictor, coarse_matched_masks)
    print(f"done, shape: {fine_matched_masks.shape}")
    del coarse_matched_masks
    del image_predictor
    for mask in fine_matched_masks:
        visualize_segmentation_mask(mask.cpu().numpy(), LF)
    return fine_matched_masks


if __name__ == "__main__":
    os.makedirs(CONFIG["files-folder"], exist_ok=True)
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = UrbanLFDataset("/home/nagonch/repos/LF_object_tracking/UrbanLF_Syn/val")
    for i, (LF, _, _) in enumerate(dataset):
        print(f"starting LF {i}")
        LF_image_sam_seg(
            mask_predictor,
            LF,
        )
        continue
        segments = LF_image_sam_seg(
            mask_predictor,
            LF,
        )
        torch.save(segments, f"segments/{str(i).zfill(4)}.pt")
        del segments
