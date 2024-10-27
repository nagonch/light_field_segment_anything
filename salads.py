from sam2_functions import get_auto_mask_predictor, generate_image_masks
from data import HCIOldDataset
import warnings
from utils import visualize_segmentation_mask, get_LF_disparities, get_mask_vis
import torch
from torchvision.transforms.functional import resize
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils import masks_to_segments

warnings.filterwarnings("ignore")

GEOM_WEIGHT = 1.0


def sort_masks(masks):
    """
    Sort [N, U, V] masks by size
    TODO: move to utils
    """
    areas = masks.sum(dim=(1, 2))
    masks = masks[torch.argsort(areas, descending=True)]
    return masks


def get_subview_masks(mask_predictor, LF):
    "[N, s, t, u, v] Get automatic masks for each LF subview"
    s_size, t_size, u_size, v_size = LF.shape[:-1]
    n_masks_min = None
    result_masks = []
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

            result_masks.append(masks)
            del masks
            print("done")
    result_masks = torch.stack([mask[:n_masks_min] for mask in result_masks]).reshape(
        n_masks_min,
        s_size,
        t_size,
        u_size,
        v_size,
    )
    return result_masks


@torch.no_grad()
def get_mask_disparities(subview_masks, disparity):
    n, s, t = subview_masks.shape[:3]
    result = torch.zeros((n,)).cuda()
    for mask_i, mask in enumerate(subview_masks):
        mask = mask[s // 2, t // 2]
        disparities_i = disparity[mask]
        result[mask_i] = disparities_i[~torch.isnan(disparities_i)].mean()
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
    mask_centroids = torch.zeros((n_masks, s_size, t_size, 2)).cuda()
    for s in range(s_size):
        for t in range(t_size):
            embedding = subview_embeddings[s, t]
            embedding = resize(embedding.permute(2, 0, 1), (u_size, v_size))
            for mask_ind in range(n_masks):
                mask_xy = torch.nonzero(subview_masks[mask_ind, s, t])
                mask_centroids[mask_ind, s, t] = mask_xy.float().mean(axis=0)
                mask_embedding = embedding[
                    :, (subview_masks[mask_ind, s, t] == 1)
                ].mean(axis=1)
                mask_embeddings[mask_ind, s, t] = mask_embedding
    print("done")
    return mask_embeddings, mask_centroids


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
def get_geometric_adjacency(LF, mask_centroids, mask_disparities):
    n_masks, s_size, t_size = mask_centroids.shape[:3]
    _, _, u_size, v_size = LF.shape[:4]
    max_point_dist = torch.norm(torch.tensor([u_size, v_size]).float())
    result = torch.zeros((n_masks, s_size, t_size, n_masks)).cuda()
    for mask_i in range(n_masks):
        mask_disparity = mask_disparities[mask_i]
        centroids_i = mask_centroids[mask_i, s_size // 2, t_size // 2]
        for s in range(s_size):
            for t in range(t_size):
                if s == s_size // 2 and t == t_size // 2:
                    continue
                centroids = mask_centroids[:, s, t]
                st = torch.tensor([s_size // 2 - s, t_size // 2 - t]).float().cuda()
                centroids_projected = centroids + mask_disparity * st
                distances = (
                    torch.norm(centroids_projected - centroids_i, dim=1)
                    / max_point_dist
                )
                distances = torch.clip(distances, min=0, max=1)
                adjacencies = 1 - distances
                result[:, s, t] = adjacencies
    return result


def optimal_matching(subview_masks, adjacency_matrix):
    adjacency_matrix = torch.clone(adjacency_matrix)
    n, s_size, t_size = adjacency_matrix.shape[:3]
    result = torch.zeros((n, s_size, t_size), dtype=torch.int32).cuda()
    for _ in range(n):
        max_idx = torch.argmax(adjacency_matrix)
        n_from, s, t, n_to = torch.unravel_index(max_idx, adjacency_matrix.shape)
        # print(adjacency_matrix[n_from, s, t, n_to], n_from, s, t, n_to)
        # plt.imshow(subview_masks[n_from, s_size // 2, t_size // 2].cpu().numpy())
        # plt.show()
        # plt.close()
        # plt.imshow(subview_masks[n_to, s, t].cpu().numpy())
        # plt.show()
        # plt.close()
        result[n_from, s, t] = n_to
        adjacency_matrix[n_from, s, t, :] = -torch.inf
        adjacency_matrix[:, :, :, n_to] = -torch.inf
    return result


def merge_masks(adjacency_matrix, match_indices, subview_masks):
    result = torch.zeros_like(subview_masks)
    n_masks, s_size, t_size = result.shape[:3]
    for mask_i in range(n_masks):
        mask = subview_masks[mask_i, s_size // 2, t_size // 2]
        result[mask_i, s_size // 2, t_size // 2] = mask
        for s in range(s_size):
            for t in range(t_size):
                if s == s_size // 2 and t == t_size // 2:
                    continue
                result[mask_i, s, t] = subview_masks[match_indices[mask_i, s, t], s, t]
    return result


def salads_LF_segmentation(mask_predictor, LF):
    "LF segmentation using greedy matching"
    # subview_masks = get_subview_masks(mask_predictor, LF)
    disparity = torch.tensor(get_LF_disparities(LF)).cuda()
    subview_masks = torch.load("subview_masks.pt")
    mask_disparities = get_mask_disparities(subview_masks, disparity)
    subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
    mask_embeddings, mask_centroids = get_mask_features(
        subview_masks, subview_embeddings
    )
    del subview_embeddings
    semantic_adjacency_matrix = get_semantic_adjacency_matrix(mask_embeddings)
    geometric_adjacency_matrix = get_geometric_adjacency(
        LF, mask_centroids, mask_disparities
    )
    adjacency_matrix = (
        semantic_adjacency_matrix * (1 - GEOM_WEIGHT)
        + geometric_adjacency_matrix * GEOM_WEIGHT
    )

    del mask_embeddings
    match_indices = optimal_matching(subview_masks, adjacency_matrix)
    result_masks = merge_masks(adjacency_matrix, match_indices, subview_masks)
    for mask in result_masks:
        vis = get_mask_vis(mask)
        plt.imshow(vis.cpu().numpy())
        plt.show()
        plt.close()
    result_segments = masks_to_segments(result_masks)
    visualize_segmentation_mask(result_segments.cpu().numpy(), LF)
    return result_masks


if __name__ == "__main__":
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = HCIOldDataset()
    for i, (LF, _, _) in enumerate(dataset):
        if i == 0:
            continue
        LF = LF[3:-3, 3:-3]
        segments = salads_LF_segmentation(
            mask_predictor,
            LF,
        )
        visualize_segmentation_mask(segments.cpu().numpy(), LF)
