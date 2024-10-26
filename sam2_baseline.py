from sam2_functions import (
    get_auto_mask_predictor,
    generate_image_masks,
    get_video_predictor,
)
from data import UrbanLFSynDataset, HCIOldDataset
import warnings
from utils import (
    visualize_segmentation_mask,
    masks_iou,
    save_LF_lawnmower,
    lawnmower_indices,
)
from time import time
import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from plenpy.lightfields import LightField
from ours import masks_to_segments

warnings.filterwarnings("ignore")
with open("sam2_baseline_LF_segmentation.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def track_masks(LF, start_masks, video_predictor):
    s, t, u, v = LF.shape[:4]
    order_indices = lawnmower_indices(s, t)
    n_masks = start_masks.shape[0]
    result = torch.zeros((n_masks, s, t, u, v), dtype=torch.bool).cuda()
    for mask_start_idx in range(0, n_masks, CONFIG["tracking-batch-size"]):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = video_predictor.init_state(CONFIG["lf-subview-folder"])
            for obj_id, mask in enumerate(
                start_masks[
                    mask_start_idx : mask_start_idx + CONFIG["tracking-batch-size"]
                ]
            ):
                video_predictor.add_new_mask(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=mask,
                )
            for (
                frame_idx,
                _,
                out_mask_logits,
            ) in video_predictor.propagate_in_video(state):
                masks_result = out_mask_logits[:, 0, :, :] > 0.0
                result[
                    mask_start_idx : mask_start_idx + CONFIG["tracking-batch-size"],
                    order_indices[frame_idx][0],
                    order_indices[frame_idx][1],
                ] = masks_result
            video_predictor.reset_state(state)
    return result


def sam2_baseline_LF_segmentation(LF, mask_predictor, video_predictor):
    start_masks = generate_image_masks(mask_predictor, LF[0, 0])
    save_LF_lawnmower(LF, CONFIG["lf-subview-folder"])
    result = track_masks(LF, start_masks, video_predictor)
    return result


def sam2_baseline_LF_segmentation_dataset(
    dataset, save_folder, continue_progress=False
):
    mask_predictor = get_auto_mask_predictor()
    video_predictor = get_video_predictor()
    time_path = f"{save_folder}/computation_times.pt"
    computation_times = []
    if continue_progress and os.path.exists(time_path):
        computation_times = torch.load(time_path).tolist()
    for i, (LF, _, _) in enumerate(dataset):
        masks_path = f"{save_folder}/{str(i).zfill(4)}_masks.pt"
        segments_path = f"{save_folder}/{str(i).zfill(4)}_segments.pt"
        if (
            all([os.path.exists(path) for path in [masks_path, segments_path]])
            and continue_progress
        ):
            continue
        start_time = time()
        result_masks = sam2_baseline_LF_segmentation(
            LF, mask_predictor, video_predictor
        )
        end_time = time()
        computation_times.append(end_time - start_time)
        result_segments = masks_to_segments(result_masks)
        torch.save(result_masks, masks_path)
        torch.save(result_segments, segments_path)
        torch.save(
            torch.tensor(computation_times),
            time_path,
        )


if __name__ == "__main__":
    pass
