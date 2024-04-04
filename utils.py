import numpy as np
import imgviz
from PIL import Image
from matplotlib import pyplot as plt
import os
import yaml
import cv2
import torch

with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def visualize_segments(segments, LF, st_border=None, filename=None):
    if st_border:
        LF = LF[st_border:-st_border, st_border:-st_border]
        segments = segments[st_border:-st_border, st_border:-st_border]
    s, t, u, v, c = LF.shape
    segments = np.transpose(segments, (0, 2, 1, 3)).reshape(s * u, t * v)
    LF = np.transpose(LF, (0, 2, 1, 3, 4)).reshape(s * u, t * v, c)
    vis = imgviz.label2rgb(
        label=segments,
        image=LF,
        colormap=imgviz.label_colormap(segments.max() + 1),
    )
    plt.imshow(vis)
    if filename:
        plt.savefig(filename)
    plt.close()


def save_LF_image(LF_image, filename="LF.jpeg", ij=None, resize_to=1024):
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
    filtering_mask = torch.ones_like(mask_u).to(torch.bool)
    for mask in [
        mask_u >= 0,
        mask_u < binary_mask.shape[0],
        mask_v >= 0,
        mask_v < binary_mask.shape[1],
    ]:
        filtering_mask = torch.logical_and(filtering_mask, mask)
    mask_u = mask_u[filtering_mask]
    mask_v = mask_v[filtering_mask]
    new_binary_mask = torch.zeros_like(binary_mask)
    new_binary_mask[mask_u, mask_v] = 1
    return new_binary_mask


if __name__ == "__main__":
    pass
