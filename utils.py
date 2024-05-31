import numpy as np
import imgviz
from PIL import Image
import yaml
import cv2
import torch
from scipy.io import savemat
from plenpy.lightfields import LightField
import logging
from k_means import k_means
from typing import Tuple

logging.getLogger("plenpy").setLevel(logging.WARNING)

with open("sam_config.yaml") as f:
    SAM_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

with open("merger_config.yaml") as f:
    MERGER_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def visualize_segmentation_mask(segments, filename=None):
    s, t, u, v = segments.shape
    segments = np.transpose(segments, (0, 2, 1, 3)).reshape(s * u, t * v)
    vis = np.transpose(
        imgviz.label2rgb(
            label=segments,
            colormap=imgviz.label_colormap(segments.max() + 1),
        ).reshape(s, u, t, v, 3),
        (0, 2, 1, 3, 4),
    )
    if filename:
        savemat(
            filename,
            {"LF": vis},
        )
    segments = LightField(vis)
    segments.show()


def visualize_segments(segments, filename):
    s, t, u, v = segments.shape
    segments = np.transpose(segments, (0, 2, 1, 3)).reshape(s * u, t * v)
    vis = imgviz.label2rgb(
        label=segments,
        colormap=imgviz.label_colormap(segments.max() + 1),
    )
    im = Image.fromarray(vis)
    im.save(filename)


def stack_segments(segments):
    s, t, u, v = segments[0].shape
    segments_result = np.zeros((s, t, u, v)).astype(np.int32)
    segment_num = 0
    for segment in segments:
        segments_result[segment] = segment_num + 1
        segment_num += 1
    segments = segments_result
    return segments_result


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1).cuda()

    return coord


def resize_LF(LF, new_u, new_v):
    s, t, u, v, _ = LF.shape
    results = []
    for s_i in range(s):
        for t_i in range(t):
            subview = Image.fromarray(LF[s_i, t_i]).resize((new_v, new_u))
            subview = np.array(subview)
            results.append(subview)
    return np.stack(results).reshape(s, t, new_u, new_v, 3)


def save_LF_image(LF_image, filename="LF.jpeg", ij=None, resize_to=None):
    if isinstance(LF_image, torch.Tensor):
        LF_image = LF_image.detach().cpu().numpy()
    LF_image = (LF_image - LF_image.min()) / (LF_image.max() - LF_image.min())
    LF_image = (LF_image * 255).astype(np.uint8)
    S, T, U, V, _ = LF_image.shape
    if ij is None:
        LF = LF_image.transpose(0, 2, 1, 3, 4).reshape(S * U, V * T, 3)
    else:
        i, j = ij
        LF = LF_image[i][j]
    im = np.array(LF)
    if resize_to is not None:
        resize_to = max(LF.shape[2], resize_to)
        if ij is None:
            resize_to *= S
        im = cv2.resize(
            im,
            (
                resize_to,
                resize_to,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    im = Image.fromarray(im)
    im.save(filename)


def get_subview_indices(s_size, t_size, remove_central=False):
    rows = torch.arange(s_size).unsqueeze(1).repeat(1, t_size).flatten()
    cols = torch.arange(t_size).repeat(s_size)

    indices = torch.stack((rows, cols), dim=-1).cuda()
    if remove_central:
        indices = torch.stack(
            [
                element
                for element in indices
                if (element != torch.tensor([s_size // 2, t_size // 2]).cuda()).any()
            ]
        )
    return indices


def get_process_to_segments_dict(
    embeddings_dict, n_processes=MERGER_CONFIG["n-parallel-processes"]
):
    segment_nums = torch.Tensor(list(embeddings_dict.keys())).cuda().long()
    classes = (
        torch.stack([torch.tensor(emb[1]) for emb in embeddings_dict.values()])
        .cuda()
        .long()
    )
    embeddings = torch.stack([emb[0] for emb in embeddings_dict.values()]).cuda()
    _, cluster_nums, _ = k_means(
        embeddings.T, classes, k=n_processes, dist="cosine", init="kmeanspp"
    )
    result_mapping = {}
    for cluster_num in torch.unique(cluster_nums):
        corresponding_segments = segment_nums[torch.where(cluster_nums == cluster_num)]
        result_mapping[cluster_num.item()] = corresponding_segments
    return result_mapping


if __name__ == "__main__":
    pass
    # segments = torch.load("merged.pt").detach().cpu().numpy()
    # visualize_segmentation_mask(segments, "segments.mat")
