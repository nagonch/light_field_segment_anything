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

warnings.filterwarnings("ignore")

with open("matching_config.yaml") as f:
    MATCHING_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


# Convert [N, U, V] masks to [U, V] segments
# The bigger the segment, the smaller the ID
# TODO: move to utils later
def reduce_masks(masks, offset):
    areas = masks.sum(dim=(1, 2))
    masks_result = torch.zeros_like(masks[0]).long()
    for i, mask_i in enumerate(torch.argsort(areas, descending=True)):
        masks_result[masks[mask_i]] = i
    masks_result[masks_result != 0] += offset
    return masks_result


# Get automatic semgents for each LF subview
def get_subview_segments(mask_predictor, LF):
    s_size, t_size, u_size, v_size = LF.shape[:-1]
    result = torch.zeros((s_size, t_size, u_size, v_size)).long().cuda()
    offset = 0
    for s in range(s_size):
        for t in range(t_size):
            subview = LF[s, t]
            masks = generate_image_masks(mask_predictor, subview).bool()
            masks = reduce_masks(
                masks,
                offset,
            )
            offset = masks.max()
            result[s][t] = masks
            del masks
    return result


# Get image embeddings for each LF subview
@torch.no_grad()
def get_subview_embeddings(predictor_model, LF):
    s_size, t_size, _, _ = LF.shape[:-1]
    results = []
    for s in range(s_size):
        for t in range(t_size):
            predictor_model.set_image(LF[s, t])
            embedding = predictor_model.get_image_embedding()
            results.append(embedding[0].permute(1, 2, 0))
    results = torch.stack(results).reshape(s_size, t_size, 64, 64, 256).cuda()
    return results


# Get embeddings for each segment
@torch.no_grad()
def get_segment_embeddings(subview_segments, subview_embeddings):
    s_size, t_size, u_size, v_size = subview_segments.shape
    segment_embeddings = {}
    for s in range(s_size):
        for t in range(t_size):
            mask = subview_segments[s, t]
            embedding = subview_embeddings[s, t]
            embedding = resize(embedding.permute(2, 0, 1), (u_size, v_size))
            for mask_ind in torch.unique(mask)[1:]:
                mask_x, mask_y = torch.where(mask == mask_ind)
                mask_embedding = embedding[:, mask_x, mask_y].mean(axis=1)
                segment_embeddings[mask_ind.item()] = mask_embedding
    return segment_embeddings


# Construct segment similarity matrix
@torch.no_grad()
def get_adjacency_matrix(subview_segments, segment_embeddings):
    s, t = subview_segments.shape[:2]
    s_reference, t_reference = s // 2, t // 2
    adjacency_inds = []
    adjacency_vals = []
    subview_segment_nums = torch.unique(subview_segments)[
        1:
    ]  # [1:] to exclude 0 (no segment)
    ref_subview_segment_nums = torch.unique(subview_segments[s_reference, t_reference])[
        1:
    ]
    n_segments = subview_segment_nums.shape[0]
    for segment_num_i in ref_subview_segment_nums:
        segment_num_i = segment_num_i.item()
        embedding_i = segment_embeddings[segment_num_i]
        embeddings_j = []
        for segment_num_j in subview_segment_nums:
            if segment_num_j in ref_subview_segment_nums:
                continue
            segment_num_j = segment_num_j.item()
            embeddings_j.append(segment_embeddings[segment_num_j])
            adjacency_inds.append(torch.tensor([segment_num_i, segment_num_j]).cuda())
        embeddings_j = torch.stack(embeddings_j)
        embeddings_i = torch.repeat_interleave(
            embedding_i[None], embeddings_j.shape[0], dim=0
        )
        similarities = F.cosine_similarity(embeddings_i, embeddings_j)
        adjacency_vals.append(similarities)
    adjacency_inds = torch.stack(adjacency_inds).T
    adjacency_vals = torch.cat(adjacency_vals, dim=0).to(torch.float32)
    adjacency_matrix = torch.sparse_coo_tensor(
        adjacency_inds, adjacency_vals, size=(n_segments, n_segments)
    )
    return adjacency_matrix


def matching_segmentation(mask_predictor, LF, filename):
    subview_segments = get_subview_segments(mask_predictor, LF)
    subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
    segment_embeddings = get_segment_embeddings(subview_segments, subview_embeddings)
    torch.save(
        subview_segments, f"{MATCHING_CONFIG['files-folder']}/{filename}_segments.pt"
    )
    torch.save(
        segment_embeddings,
        f"{MATCHING_CONFIG['files-folder']}/{filename}_embeddings.pt",
    )
    adj_matrix = get_adjacency_matrix(subview_segments, segment_embeddings).to_dense()
    torch.save(
        adj_matrix,
        f"{MATCHING_CONFIG['files-folder']}/{filename}_adj_matrix.pt",
    )
    plt.imshow(adj_matrix.cpu().numpy(), cmap="gray")
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.makedirs(MATCHING_CONFIG["files-folder"], exist_ok=True)
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = HCIOldDataset()
    for i, (LF, _, _) in enumerate(dataset):
        LF = LF[3:-3, 3:-3]
        matching_segmentation(
            mask_predictor,
            LF,
            filename=str(i).zfill(4),
        )
        raise
