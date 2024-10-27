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
from salads2 import LF_segment_merger

warnings.filterwarnings("ignore")

GEOM_WEIGHT = 0.5
CERTAINTY_THRESH = 0.8


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
        result[n_from, s_best, t_best] = n_to
        adjacency_matrix[n_from, s_best, t_best, :] = -torch.inf
        adjacency_matrix[:, s_best, t_best, n_to] = -torch.inf
        for n_subview in range(s_size * t_size - 2):
            max_idx = torch.argmax(adjacency_matrix[n_from])
            s, t, n_to = torch.unravel_index(max_idx, adjacency_matrix.shape[1:])
            result[n_from, s, t] = (
                n_to if adjacency_matrix[n_from, s, t, n_to] >= CERTAINTY_THRESH else -1
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


def convert_masks(subview_masks):
    n_masks, s_size, t_size, u_size, v_size = subview_masks.shape
    result = torch.zeros((s_size, t_size, u_size, v_size)).cuda()
    i_mask = 1
    for mask_i in range(n_masks):
        for s in range(s_size):
            for t in range(t_size):
                result[s, t][subview_masks[mask_i, s, t]] += i_mask
                i_mask += 1
    return result


def filter_masks(subview_masks):
    s, t, u, v = subview_masks.shape
    for i in torch.unique(subview_masks)[1:]:
        if (subview_masks == i).sum() <= u * v * 0.01:
            subview_masks[subview_masks == i] = 0
    return subview_masks


def get_segment_embeddings(subview_segments, subview_embeddings):
    result_embeddings = {}
    s_size, t_size, u_size, v_size = subview_segments.shape[:4]
    for i in torch.unique(subview_segments)[1:]:
        mask = (subview_segments == i).sum(axis=(0, 1))
        s, t = torch.unravel_index(
            torch.argmax((subview_segments == i).sum(axis=(2, 3))), (s_size, t_size)
        )
        embedding = subview_embeddings[s, t]
        embedding = resize(embedding.permute(2, 0, 1), (u_size, v_size))
        x, y = torch.where(mask == 1)
        result_embeddings[i.item()] = (embedding[:, x, y].mean(axis=1), None)
    return result_embeddings


def salads_LF_segmentation(mask_predictor, LF):
    "LF segmentation using greedy matching"
    # subview_masks = get_subview_masks(mask_predictor, LF)
    subview_masks = torch.load("subview_masks.pt")
    subview_segments = convert_masks(subview_masks)
    subview_segments = filter_masks(subview_segments)
    subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
    embeddings = get_segment_embeddings(subview_segments, subview_embeddings)
    merger = LF_segment_merger(subview_segments, embeddings, LF)
    segments = merger.get_result_masks().long()
    # print(segments, torch.unique(segments))
    # for mask_i in torch.unique(segments)[1:]:
    #     plt.imshow(get_mask_vis(segments == mask_i).cpu().numpy())
    #     plt.show()
    #     plt.close()
    visualize_segmentation_mask(segments.cpu().numpy(), LF)
    # disparity = torch.tensor(get_LF_disparities(LF)).cuda()
    # mask_disparities = get_mask_disparities(subview_masks, disparity)
    # subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
    # mask_embeddings, mask_centroids = get_mask_features(
    #     subview_masks, subview_embeddings
    # )
    # del subview_embeddings
    # semantic_adjacency_matrix = get_semantic_adjacency_matrix(mask_embeddings)
    # geometric_adjacency_matrix = get_geometric_adjacency(
    #     LF, mask_centroids, mask_disparities
    # )
    # adjacency_matrix = (
    #     semantic_adjacency_matrix * (1 - GEOM_WEIGHT)
    #     + geometric_adjacency_matrix * GEOM_WEIGHT
    # )

    # del mask_embeddings
    # match_indices, order = optimal_matching(adjacency_matrix)
    # result_masks = merge_masks(match_indices, subview_masks)[order]
    # torch.save(result_masks, "result_masks.pt")
    # for mask in result_masks:
    #     plt.imshow(get_mask_vis(mask).cpu().numpy())
    #     plt.show()
    #     plt.close()
    # raise
    # result_segments = masks_to_segments(result_masks)
    # visualize_segmentation_mask(result_segments.cpu().numpy(), LF)
    # return result_masks


if __name__ == "__main__":
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = HCIOldDataset()
    for i, (LF, _, _) in enumerate(dataset):
        if i == 0:
            continue
        LF = LF[2:-2, 2:-2]
        segments = salads_LF_segmentation(
            mask_predictor,
            LF,
        )
        visualize_segmentation_mask(segments.cpu().numpy(), LF)
