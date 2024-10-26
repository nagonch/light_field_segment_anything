from sam2_functions import get_auto_mask_predictor, generate_image_masks
from data import HCIOldDataset
import warnings
from utils import visualize_segmentation_mask
import torch
from torchvision.transforms.functional import resize
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import masks_to_segments

warnings.filterwarnings("ignore")


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
                mask_embedding = embedding[:, mask_xy[0], mask_xy[1]].mean(axis=1)
                mask_embeddings[mask_ind, s, t] = mask_embedding
    print("done")
    return mask_embeddings, mask_centroids


@torch.no_grad()
def get_sim_adjacency_matrix(subview_embeddings):
    """
    [n, s, t, n] Get mask cosine similarities matrix
    Logic: [mask_from, s, t, mask_to]
    """
    n_masks, s_size, t_size = subview_embeddings.shape[:3]
    result = torch.zeros((n_masks, s_size, t_size, n_masks)).cuda()
    for mask_i in range(n_masks):
        mask_embedding = torch.repeat_interleave(
            subview_embeddings[mask_i, s_size // 2, t_size // 2][None], n_masks, dim=0
        )
        for s in range(s_size):
            for t in range(t_size):
                if s == s_size // 2 and t == t_size // 2:
                    continue
                embeddings = subview_embeddings[:, s, t]
                sim = F.cosine_similarity(mask_embedding, embeddings)
                result[mask_i, s, t, :] = sim
    return result


# @torch.no_grad()
# def get_sim_adjacency_matrix(subview_segments, segment_embeddings):
#     "Construct segment similarity matrix"
#     print("getting adjacency matrix...", end="")
#     s, t = subview_segments.shape[:2]
#     s_reference, t_reference = s // 2, t // 2
#     adjacency_inds = []
#     adjacency_vals = []
#     subview_segment_nums = torch.unique(subview_segments)[
#         1:
#     ]  # [1:] to exclude 0 (no segment)
#     ref_subview_segment_nums = torch.unique(subview_segments[s_reference, t_reference])[
#         1:
#     ]
#     n_segments = subview_segment_nums.shape[0]
#     for segment_num_i in ref_subview_segment_nums:
#         segment_num_i = segment_num_i.item()
#         embedding_i = segment_embeddings[segment_num_i]
#         embeddings_j = []
#         for segment_num_j in subview_segment_nums:
#             if segment_num_j in ref_subview_segment_nums:
#                 continue
#             segment_num_j = segment_num_j.item()
#             embeddings_j.append(segment_embeddings[segment_num_j])
#             adjacency_inds.append(torch.tensor([segment_num_i, segment_num_j]).cuda())
#         embeddings_j = torch.stack(embeddings_j)
#         embeddings_i = torch.repeat_interleave(
#             embedding_i[None], embeddings_j.shape[0], dim=0
#         )
#         similarities = F.cosine_similarity(embeddings_i, embeddings_j)
#         adjacency_vals.append(similarities)
#     adjacency_inds = torch.stack(adjacency_inds).T.cuda()
#     adjacency_vals = torch.cat(adjacency_vals, dim=0).to(torch.float32).cuda()
#     adjacency_matrix = torch.sparse_coo_tensor(
#         adjacency_inds, adjacency_vals, size=(n_segments + 1, n_segments + 1)
#     ).to_dense()
#     print("done")
#     return adjacency_matrix


def greedy_matching(subview_segments, sim_adjacency_matrix):
    s_size, t_size = subview_segments.shape[:2]
    s_reference, t_reference = s_size // 2, t_size // 2
    ref_segment_nums = torch.unique(subview_segments[s_reference, t_reference])[
        1:
    ]  # exclude 0 (no segment)
    for segment_i in ref_segment_nums:
        for s in range(s_size):
            for t in range(t_size):
                if s == s_reference and t == t_reference:
                    continue
                st_unique_segments = torch.unique(subview_segments[s, t])[1:]
                if len(st_unique_segments) == 0:
                    continue  # nothing left to match with in this subview
                similarities = sim_adjacency_matrix[segment_i, st_unique_segments]
                match_segment_id = st_unique_segments[torch.argmax(similarities)]
                subview_segments[subview_segments == match_segment_id] = segment_i
    return subview_segments


def salads_LF_segmentation(mask_predictor, LF):
    "LF segmentation using greedy matching"
    # subview_masks = get_subview_masks(mask_predictor, LF)
    # subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
    subview_masks = torch.load("subview_masks.pt")
    subview_embeddings = torch.load("subview_embeddings.pt")
    mask_embeddings, mask_centroids = get_mask_features(
        subview_masks, subview_embeddings
    )
    del subview_embeddings
    sim_adjacency_matrix = get_sim_adjacency_matrix(mask_embeddings)
    print(sim_adjacency_matrix.shape)
    raise
    # del segment_embeddings
    # matched_segments = greedy_matching(subview_masks, sim_adjacency_matrix)
    # return matched_segments


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
