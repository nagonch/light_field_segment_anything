from sam2_functions import (
    get_auto_mask_predictor,
    get_sam_1_auto_mask_predictor,
    generate_image_masks,
)
from data import HCIOldDataset, UrbanLFSynDataset
import warnings
from utils import (
    visualize_segmentation_mask,
    masks_iou,
    masks_to_segments,
    predict_mask_subview_position,
)
from time import time
import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from utils import get_LF_disparities

warnings.filterwarnings("ignore")

with open("ours.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def get_mask_disparities(masks_central, disparities):
    """
    Get mean disparity of each mask
    masks_central: torch.tensor [n, u, v] (torch.bool)
    disparities: np.array [u, v] (np.float32)
    returns: torch.tensor [n] (torch.float32)
    """
    mask_disparities = torch.zeros((masks_central.shape[0],)).cuda()
    for i, mask_i in enumerate(masks_central):
        disparities_i = disparities[mask_i]
        disparities_i = disparities_i[~torch.any(disparities_i.isnan())]
        mask_disparities[i] = torch.median(disparities[mask_i]).item()
    return mask_disparities


@torch.no_grad()
def get_subview_embeddings(predictor_model, LF):
    "[s, t, 64, 64, 256] Get image embeddings for each LF subview"
    print("getting subview embeddings...", end="")
    s_size, t_size, _, _ = LF.shape[:-1]
    results = []
    for s in range(s_size):
        for t in range(t_size):
            predictor_model.set_image(LF[s, t])
            embedding = predictor_model.get_image_embedding()
            results.append(embedding[0].permute(1, 2, 0))
    results = torch.stack(results).reshape(s_size, t_size, 64, 64, 256).cuda()
    print("done")
    return results


def get_coarse_matching(LF, masks_central, mask_disparities, disparities):
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
    for s in range(s_size):
        for t in range(t_size):
            for i, (mask, disparity) in enumerate(zip(masks_central, mask_disparities)):
                result[i][s][t] = predict_mask_subview_position(
                    mask, disparities, s - s_size // 2, t - t_size // 2
                )
    return result


@torch.no_grad()
def refine_coarse_masks_semantic(
    subview_embeddings,
    coarse_masks,
):
    n_masks, s_size, t_size, u_size, v_size = coarse_masks.shape
    coarse_masks = coarse_masks.to(torch.float16)
    for mask_i in range(n_masks):
        mask = coarse_masks[mask_i, s_size // 2, t_size // 2]
        embedding = subview_embeddings[s_size // 2, t_size // 2]
        embedding = resize(embedding.permute(2, 0, 1), (u_size, v_size))
        mask_embedding = embedding[:, (mask == 1)].mean(axis=1)
        for s in range(s_size):
            for t in range(t_size):
                if s == s_size // 2 and t == t_size // 2:
                    continue
                mask_st = coarse_masks[mask_i, s, t]
                embeddings_st = subview_embeddings[s, t]
                embeddings_st = resize(embeddings_st.permute(2, 0, 1), (u_size, v_size))
                embeddings_st = embeddings_st[:, mask_st == 1]
                similarities = F.cosine_similarity(
                    embeddings_st.T, mask_embedding[:, None].T
                )
                similarities = (
                    similarities * (similarities > CONFIG["sim-thresh"]).float()
                )
                coarse_masks[mask_i, s, t][mask_st == 1] = similarities.to(
                    torch.float16
                )
                del similarities
                del embeddings_st
                del mask_st
    return coarse_masks


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
            for mask_i, mask in enumerate(coarse_masks[:, s, t]):
                point_prompts_i = torch.nonzero(mask).flip(1)
                if point_prompts_i.shape[0] == 0:
                    continue
                box_pormpts_i = torch.tensor(
                    [
                        point_prompts_i[:, 0].min(),
                        point_prompts_i[:, 1].min(),
                        point_prompts_i[:, 0].max(),
                        point_prompts_i[:, 1].max(),
                    ]
                ).cuda()
                if CONFIG["use-semantic"]:
                    weights = mask[point_prompts_i[:, 1], point_prompts_i[:, 0]].float()
                    point_prompts_i_centroid = (
                        point_prompts_i.float().T @ weights[:, None]
                    ) / weights.sum()
                    point_prompts_i_centroid = point_prompts_i_centroid[:, 0]
                else:
                    point_prompts_i_centroid = point_prompts_i.float().mean(axis=0)
                distances = torch.norm(
                    point_prompts_i - point_prompts_i_centroid, dim=1
                )
                point_prompts_i = point_prompts_i[torch.argmin(distances), :][None]
                point_prompts[mask_i, s, t] = point_prompts_i
                box_prompts[mask_i, s, t] = box_pormpts_i
    return point_prompts, box_prompts


def get_refined_matching(LF, image_predictor, coarse_masks, point_prompts, box_prompts):
    """
    Predict subview masks using disparities
    LF: np.array [s, t, u, v, 3] (np.uint8)
    image_predictor: SAM2ImagePredictor
    coarse_masks: torch.tensor [n, s, t, u, v] (torch.bool)
    returns: torch.tensor [n, s, t, u, v] (torch.bool)
    """
    s_size, t_size = LF.shape[:2]
    n = coarse_masks.shape[0]
    for s in range(s_size):
        for t in range(t_size):
            if s == s_size // 2 and t == t_size // 2:
                continue
            coarse_masks_st = torch.clone(coarse_masks[:, s, t, :, :])
            image_predictor.set_image(LF[s, t])
            point_prompts_st = point_prompts[:, s, t]
            box_prompts_st = box_prompts[:, s, t]
            for segment_i, (point_prompts_i, box_prompts_i) in enumerate(
                zip(point_prompts_st, box_prompts_st)
            ):
                point_prompts_i = point_prompts_i[None]
                if point_prompts_i.sum() <= 1e-6:
                    continue
                labels = torch.ones(point_prompts_i.shape[0])
                fine_segment_result, _, _ = image_predictor.predict(
                    point_coords=point_prompts_i,
                    point_labels=labels,
                    box=box_prompts_i,
                    multimask_output=True,
                )
                fine_segment_result = torch.tensor(
                    fine_segment_result, dtype=torch.bool
                ).cuda()
                ious = masks_iou(fine_segment_result, coarse_masks_st[segment_i])
                if ious.max() > CONFIG["iou-thresh"]:
                    match_idx = torch.argmax(ious)
                    coarse_masks[segment_i, s, t] = fine_segment_result[
                        match_idx
                    ]  # replacing coarse masks with fine ones
    return coarse_masks


def filter_final_masks(masks, relative_area_min=CONFIG["relative-min-area"]):
    result = []
    _, _, _, u, v = masks.shape
    for mask in masks:
        if mask.sum(dim=(2, 3)).float().mean() / float(u * v) >= relative_area_min:
            result.append(mask)
    return torch.stack(result)


def sam_fast_LF_segmentation(mask_predictor, LF, visualize=False):
    s_central, t_central = LF.shape[0] // 2, LF.shape[1] // 2

    print("generate_image_masks...", end="")
    masks_central = generate_image_masks(mask_predictor, LF[s_central, t_central])
    print(f"done, shape: {masks_central.shape}")

    print("get_LF_disparities...", end="")
    disparities = torch.tensor(get_LF_disparities(LF)).cuda()
    print(f"done, shape: {disparities.shape}")

    print("get_mask_disparities...", end="")
    mask_disparities = get_mask_disparities(masks_central, disparities)
    mask_depth_order = torch.argsort(mask_disparities)
    masks_central = masks_central[mask_depth_order]
    mask_disparities = mask_disparities[mask_depth_order]
    del mask_depth_order
    print(f"done, shape: {mask_disparities.shape}")
    print("get_coarse_matching...", end="")
    coarse_matched_masks = get_coarse_matching(
        LF, masks_central, mask_disparities, disparities
    )
    print(f"done, shape: {coarse_matched_masks.shape}")
    del mask_disparities
    del masks_central
    del disparities
    if CONFIG["use-semantic"]:
        subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
        weighted_coarse_masks = refine_coarse_masks_semantic(
            subview_embeddings, coarse_matched_masks
        )
        del subview_embeddings
        point_prompts, box_prompts = get_prompts_for_masks(weighted_coarse_masks)
        del weighted_coarse_masks
    else:
        point_prompts, box_prompts = get_prompts_for_masks(coarse_matched_masks)
    print("get_fine_matching...", end="")
    refined_matched_masks = get_refined_matching(
        LF, mask_predictor.predictor, coarse_matched_masks, point_prompts, box_prompts
    )
    print(f"done, shape: {refined_matched_masks.shape}")
    del mask_predictor
    del coarse_matched_masks
    if visualize:
        print("visualizing segments...")
        refined_segments = masks_to_segments(refined_matched_masks)
        visualize_segmentation_mask(refined_segments.cpu().numpy())
    return refined_matched_masks


def sam_fast_LF_segmentation_dataset(
    dataset,
    save_folder,
    continue_progress=False,
    visualize=False,
):
    mask_predictor = (
        get_auto_mask_predictor()
        if CONFIG["sam-version"] == 2
        else get_sam_1_auto_mask_predictor()
    )
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
        print(f"segmenting lf {i}")
        start_time = time()
        result_masks = sam_fast_LF_segmentation(
            mask_predictor,
            LF,
            visualize=visualize,
        )
        end_time = time()
        computation_times.append(
            (end_time - start_time)
            / float(
                result_masks.shape[0] * result_masks.shape[1] * result_masks.shape[2]
            )
        )
        result_segments = masks_to_segments(result_masks)
        torch.save(result_masks, masks_path)
        torch.save(result_segments, segments_path)
        del result_masks
        del result_segments
        torch.save(
            torch.tensor(computation_times),
            time_path,
        )


if __name__ == "__main__":
    dataset = UrbanLFSynDataset("UrbanLF_Syn/val")
    sam_fast_LF_segmentation_dataset(dataset, "test_result", visualize=True)
