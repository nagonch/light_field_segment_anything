import numpy as np
import imgviz
from PIL import Image
from matplotlib import cm, pyplot as plt
import os
import yaml
import cv2
import torch

with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def visualize_segments(segments, st_border=None, filename=None):
    if st_border:
        segments = segments[st_border:-st_border, st_border:-st_border]
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


def project_point_onto_line(x, v, y):
    # Calculate the vector from x to y
    a = y - x

    # Calculate the projection of a onto v
    projection = torch.dot(a, v) / torch.dot(v, v) * v

    # Calculate the scalar t
    t = torch.dot(projection, v) / torch.dot(v, v)

    return t


def draw_line_in_mask(mask, start_point, end_point):
    # Extract x and y coordinates of start and end points
    x0, y0 = start_point[0], start_point[1]
    x1, y1 = end_point[0], end_point[1]

    # Compute differences between start and end points
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # Determine direction of the line
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Compute the initial error
    error = dx - dy

    # Initialize the current position to the start point
    x, y = x0, y0

    # Loop until we reach the end point
    while x != x1 or y != y1:
        # Append the current position to the list of points
        mask[y, x] = 1

        # Compute the error for the next position
        e2 = 2 * error

        # Determine which direction to move
        if e2 > -dy:
            error = error - dy
            x = x + sx
        if e2 < dx:
            error = error + dx
            y = y + sy

    # Append the final position to the list of points
    mask[y1, x1] = 1

    return mask


def line_image_boundaries(P, V, M, N):
    Px, Py = P
    Vx, Vy = V

    t_left = -Px / Vx if Vx != 0 else float("inf")
    t_right = (N - 1 - Px) / Vx if Vx != 0 else float("inf")
    t_top = -Py / Vy if Vy != 0 else float("inf")
    t_bottom = (M - 1 - Py) / Vy if Vy != 0 else float("inf")

    Q_left = (Px + t_left * Vx, Py + t_left * Vy)
    Q_right = (Px + t_right * Vx, Py + t_right * Vy)
    Q_top = (Px + t_top * Vx, Py + t_top * Vy)
    Q_bottom = (Px + t_bottom * Vx, Py + t_bottom * Vy)

    points = [Q_left, Q_right, Q_top, Q_bottom]
    valid_points = [(int(x), int(y)) for (x, y) in points if 0 <= x < N and 0 <= y < M]

    return list(set(valid_points))


if __name__ == "__main__":
    pass
