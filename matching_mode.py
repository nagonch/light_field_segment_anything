from sam2_functions import get_auto_mask_predictor, generate_image_masks
from data import HCIOldDataset
import warnings
from utils import visualize_segmentation_mask
import torch
from torchvision.transforms.functional import resize
import numpy as np

warnings.filterwarnings("ignore")


# move to utils later
def reduce_masks(masks, offset):
    areas = masks.sum(dim=(1, 2))
    masks_result = torch.zeros_like(masks[0]).long()
    for i, _ in enumerate(torch.argsort(areas, descending=True)):
        masks_result[masks[i]] += i
        masks_result = torch.clip(masks_result, 0, i).long()
    masks_result[masks_result != 0] += offset
    return masks_result


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


@torch.no_grad()
def get_mask_embeddings(subview_segments, subview_embeddings):
    s_size, t_size, u_size, v_size = subview_segments.shape
    mask_embeddings = {}
    for s in range(s_size):
        for t in range(t_size):
            mask = subview_segments[s, t]
            embedding = subview_embeddings[s, t]
            embedding = resize(embedding.permute(2, 0, 1), (u_size, v_size))
            for mask_ind in torch.unique(mask)[1:]:
                mask_x, mask_y = torch.where(mask == mask_ind)
                mask_embedding = embedding[:, mask_x, mask_y].mean(axis=1)
                mask_embeddings[mask_ind.item()] = mask_embedding
    return mask_embeddings


def matching_segmentation(mask_predictor, LF):
    subview_segments = get_subview_segments(mask_predictor, LF)
    subview_embeddings = get_subview_embeddings(mask_predictor.predictor, LF)
    # torch.save(subview_embeddings, "subview_embeddings.pt")
    # torch.save(subview_segments, "subview_segments.pt")
    # subview_embeddings = torch.load("subview_embeddings.pt")
    # subview_segments = torch.load("subview_segments.pt")
    mask_embeddings = get_mask_embeddings(subview_segments, subview_embeddings)


if __name__ == "__main__":
    mask_predictor = get_auto_mask_predictor()
    image_predictor = mask_predictor.predictor
    dataset = HCIOldDataset()
    for LF, _, _ in dataset:
        LF = LF[3:-3, 3:-3]
        matching_segmentation(mask_predictor, LF)
