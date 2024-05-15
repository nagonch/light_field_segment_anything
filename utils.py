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

logging.getLogger("plenpy").setLevel(logging.WARNING)

with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def visualize_segmentation_mask(segments, filename):
    s, t, u, v = segments.shape
    segments = np.transpose(segments, (0, 2, 1, 3)).reshape(s * u, t * v)
    vis = np.transpose(
        imgviz.label2rgb(
            label=segments,
            colormap=imgviz.label_colormap(segments.max() + 1),
        ).reshape(s, u, t, v, 3),
        (0, 2, 1, 3, 4),
    )
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


def shift_binary_mask(binary_mask, uv_shift):
    mask_u, mask_v = torch.where(binary_mask == 1)
    mask_u += uv_shift[0]
    mask_v += uv_shift[1]
    filtering_mask = torch.ones_like(mask_u).to(torch.bool).cuda()
    for mask in [
        mask_u >= 0,
        mask_u < binary_mask.shape[0],
        mask_v >= 0,
        mask_v < binary_mask.shape[1],
    ]:
        filtering_mask = torch.logical_and(filtering_mask, mask)
    mask_u = mask_u[filtering_mask]
    mask_v = mask_v[filtering_mask]
    new_binary_mask = torch.zeros_like(binary_mask).cuda()
    new_binary_mask[mask_u, mask_v] = 1
    return new_binary_mask


def project_point_onto_line(x, v, y):
    # Calculate the vector from x to y
    a = y - x

    # Calculate the projection of a onto v
    projection = torch.dot(a, v) / torch.dot(v, v) * v

    # Calculate the scalar t
    t = torch.dot(projection, v) / torch.dot(v, v)

    return t


def test_mask(mask, p, v):
    v_len = torch.norm(v)
    u_mask, v_mask = torch.where(mask == 1)
    error = torch.abs(v[0] * (u_mask - p[0]) + v[1] * (v_mask - p[1])) / v_len
    return error.min() <= CONFIG["mask-test-threshold"]


def binary_mask_centroid(mask):
    nonzero_indices = torch.nonzero(mask)
    centroid = nonzero_indices.float().mean(axis=0)
    return centroid


def get_subview_indices(s_size, t_size):
    rows = torch.arange(s_size).unsqueeze(1).repeat(1, t_size).flatten()
    cols = torch.arange(t_size).repeat(s_size)

    indices = torch.stack((rows, cols), dim=-1).cuda()
    return indices


def get_process_to_segments_dict(
    embeddings_filename, n_processes=CONFIG["n-parallel-processes"]
):
    embeddings_dict = torch.load(embeddings_filename)
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


def calculate_outliers(float_tensor):
    quantile1 = torch.quantile(float_tensor, 0.99)
    quantile2 = torch.quantile(float_tensor, 0.1)
    n_outliers = ((float_tensor >= quantile1) & (float_tensor <= quantile2)).sum()
    return n_outliers


if __name__ == "__main__":
    segments = torch.load("merged.pt").detach().cpu().numpy()
    visualize_segmentation_mask(segments, "segments.mat")
