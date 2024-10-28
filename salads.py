from sam2_functions import get_sam_1_auto_mask_predictor, generate_image_masks
from data import HCIOldDataset, UrbanLFSynDataset
import warnings
from utils import visualize_segmentation_mask, get_LF_disparities
import torch
from torchvision.transforms.functional import resize
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import predict_mask_subview_position, masks_iou, masks_to_segments
import yaml
from time import time
import os


warnings.filterwarnings("ignore")
with open("salads_config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def sort_masks(masks):
    """
    Sort [N, U, V] masks by size
    TODO: move to utils
    """
    areas = masks.sum(dim=(1, 2))
    masks = masks[torch.argsort(areas, descending=True)]
    return masks


def stack_segments(segments):
    s, t, u, v = segments[0].shape
    segments_result = np.zeros((s, t, u, v)).astype(np.int32)
    segment_num = 0
    for segment in segments:
        segments_result[segment] = segment_num + 1
        segment_num += 1
    segments = segments_result
    return segments_result


def segment_subviews(mask_predictor, LF):
    "[N, s, t, u, v] Get automatic masks for each LF subview"
    s_size, t_size, _, _ = LF.shape[:-1]
    n_masks_min = None
    for s in range(s_size):
        for t in range(t_size):
            print(f"getting masks for subview {s, t}...", end="")
            subview = LF[s, t]
            masks = generate_image_masks(mask_predictor, subview).bool().cuda()
            masks = sort_masks(
                masks,
            )
            n_masks_min = (
                min(n_masks_min, masks.shape[0]) if n_masks_min else masks.shape[0]
            )
            torch.save(masks, f'{CONFIG["tmp-folder"]}/{s}_{t}.pt')
            del masks
            print("done")
    return n_masks_min


def gather_masks(LF, n_masks_min):
    s_size, t_size, u_size, v_size = LF.shape[:4]
    result = torch.zeros(
        (n_masks_min, s_size, t_size, u_size, v_size), dtype=torch.bool, device="cuda"
    )
    for s in range(s_size):
        for t in range(t_size):
            result[:, s, t] = torch.load(f'{CONFIG["tmp-folder"]}/{s}_{t}.pt')[
                :n_masks_min
            ]
    return result


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


@torch.no_grad()
def get_mask_features(subview_masks, subview_embeddings):
    "[n, s, t, 256], [n, s, t, 3] Get embeddings and centroids for each mask"
    print("getting mask embeddings...", end="")
    n_masks, s_size, t_size, u_size, v_size = subview_masks.shape
    mask_embeddings = torch.zeros((n_masks, s_size, t_size, 256)).cuda()
    for s in range(s_size):
        for t in range(t_size):
            embedding = subview_embeddings[s, t]
            embedding = resize(embedding.permute(2, 0, 1), (u_size, v_size))
            for mask_ind in range(n_masks):
                mask_embedding = embedding[
                    :, (subview_masks[mask_ind, s, t] == 1)
                ].mean(axis=1)
                mask_embeddings[mask_ind, s, t] = mask_embedding
    print("done")
    return mask_embeddings


@torch.no_grad()
def get_semantic_adjacency_matrix(mask_embeddings):
    """
    [n, s, t, n] Get mask cosine similarities matrix
    Logic: [mask_from, s, t, mask_to]
    """
    n_masks, s_size, t_size = mask_embeddings.shape[:3]
    result = torch.zeros((n_masks, s_size, t_size, n_masks)).cuda()
    for mask_i in range(n_masks):
        mask_embedding = torch.repeat_interleave(
            mask_embeddings[mask_i, s_size // 2, t_size // 2][None], n_masks, dim=0
        )
        for s in range(s_size):
            for t in range(t_size):
                if s == s_size // 2 and t == t_size // 2:
                    continue
                embeddings = mask_embeddings[:, s, t]
                sim = F.cosine_similarity(mask_embedding, embeddings)
                result[mask_i, s, t, :] = sim
    return result


@torch.no_grad()
def get_geometric_adjacency(LF, masks, disparity):
    n_masks, s_size, t_size = masks.shape[:3]
    _, _, u_size, v_size = LF.shape[:4]
    result = torch.zeros((n_masks, s_size, t_size, n_masks), dtype=torch.float64).cuda()
    for mask_i in range(n_masks):
        mask_target = masks[mask_i, s_size // 2, t_size // 2]
        for s in range(s_size):
            for t in range(t_size):
                if s == s_size // 2 and t == t_size // 2:
                    continue
                mask_shifted = predict_mask_subview_position(
                    mask_target, disparity, s - s_size // 2, t - t_size // 2
                )
                ious = masks_iou(masks[:, s, t], mask_shifted)
                result[mask_i, s, t, :] = ious
    return result


def optimal_matching(adjacency_matrix):
    adjacency_matrix = torch.clone(adjacency_matrix)
    n, s_size, t_size = adjacency_matrix.shape[:3]
    result = torch.zeros((n, s_size, t_size), dtype=torch.int32).cuda()
    order = []
    for mask_i in range(n):
        max_idx = torch.argmax(adjacency_matrix)
        # find best match globaly
        n_from, s_best, t_best, n_to = torch.unravel_index(
            max_idx, adjacency_matrix.shape
        )
        cert_threshold = torch.clone(adjacency_matrix[n_from, s_best, t_best, n_to])
        result[n_from, s_best, t_best] = n_to
        adjacency_matrix[n_from, s_best, t_best, :] = -torch.inf
        adjacency_matrix[:, s_best, t_best, n_to] = -torch.inf
        for n_subview in range(s_size * t_size - 2):
            max_idx = torch.argmax(adjacency_matrix[n_from])
            s, t, n_to = torch.unravel_index(max_idx, adjacency_matrix.shape[1:])
            result[n_from, s, t] = (
                n_to
                if adjacency_matrix[n_from, s, t, n_to]
                >= (cert_threshold - CONFIG["thresh-tolerance"])
                else -1
            )
            adjacency_matrix[n_from, s, t, :] = -torch.inf
            adjacency_matrix[:, s, t, n_to] = -torch.inf
        adjacency_matrix[n_from, :, :, :] = -torch.inf
        order.append(torch.tensor(mask_i))
    return result, torch.stack(order)


def merge_masks(match_indices, subview_masks):
    result = torch.zeros_like(subview_masks)
    n_masks, s_size, t_size = result.shape[:3]
    for mask_i in range(n_masks):
        mask = subview_masks[mask_i, s_size // 2, t_size // 2]
        result[mask_i, s_size // 2, t_size // 2] = mask
        for s in range(s_size):
            for t in range(t_size):
                if s == s_size // 2 and t == t_size // 2:
                    continue
                if match_indices[mask_i, s, t] >= 0:
                    result[mask_i, s, t] = subview_masks[
                        match_indices[mask_i, s, t], s, t
                    ]
    return result


def salads_LF_segmentation(LF):
    mask_predictor = get_sam_1_auto_mask_predictor()
    "LF segmentation using greedy matching"
    n_masks_min = segment_subviews(mask_predictor, LF)
    subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
    del mask_predictor
    subview_masks = gather_masks(LF, n_masks_min)
    disparity = torch.tensor(get_LF_disparities(LF)).cuda()
    mask_embeddings = get_mask_features(subview_masks, subview_embeddings)
    del subview_embeddings
    semantic_adjacency_matrix = get_semantic_adjacency_matrix(mask_embeddings)
    geometric_adjacency_matrix = get_geometric_adjacency(LF, subview_masks, disparity)
    adjacency_matrix = (
        semantic_adjacency_matrix * (1 - CONFIG["geom-weight"])
        + geometric_adjacency_matrix * CONFIG["geom-weight"]
    )

    del mask_embeddings
    match_indices, order = optimal_matching(adjacency_matrix)
    result_masks = merge_masks(match_indices, subview_masks)[order]
    return result_masks


def salads_LF_segmentation_dataset(
    dataset, save_folder, continue_progress=False, visualize=False
):
    os.makedirs(CONFIG["tmp-folder"], exist_ok=True)
    time_path = f"{save_folder}/computation_times.pt"
    computation_times = []
    if continue_progress and os.path.exists(time_path):
        computation_times = torch.load(time_path).tolist()
    for i, (LF, _, _) in enumerate(dataset):
        try:
            masks_path = f"{save_folder}/{str(i).zfill(4)}_masks.pt"
            segments_path = f"{save_folder}/{str(i).zfill(4)}_segments.pt"
            if (
                all([os.path.exists(path) for path in [masks_path, segments_path]])
                and continue_progress
            ):
                continue
            start_time = time()
            result_masks = salads_LF_segmentation(LF)
            end_time = time()
            computation_times.append(end_time - start_time)
            result_segments = masks_to_segments(result_masks)
            if visualize:
                visualize_segmentation_mask(result_segments.cpu().numpy())
            torch.save(result_masks, masks_path)
            torch.save(result_segments, segments_path)
            torch.save(
                torch.tensor(computation_times),
                time_path,
            )
        except RuntimeError as e:
            print(f"\n{i} ERROR: {e}")
            continue


if __name__ == "__main__":
    dataset = UrbanLFSynDataset("UrbanLF_Syn/val")
    salads_LF_segmentation_dataset(dataset, "salads_test", visualize=True)
