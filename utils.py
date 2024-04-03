import numpy as np
import imgviz
from PIL import Image
from matplotlib import pyplot as plt
import os
import yaml
import cv2

with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def get_LF(dir):
    subviews = []
    u = v = CONFIG["lf-subview-max-size"]
    for img in list(sorted(os.listdir(dir))):
        path = dir + "/" + img
        subviews.append(np.array(Image.open(path).resize((u, v)))[:, :, :3])
    n_apertures = int(np.sqrt(len(subviews)))
    u, v, c = subviews[0].shape
    LF = np.stack(subviews).reshape(n_apertures, n_apertures, u, v, c).astype(np.uint8)

    return LF


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
    LF_image = (LF_image - LF_image.min()) / (LF_image.max() - LF_image.min())
    LF_image = (LF_image * 255).astype(np.uint8)
    N, N, M, M, _ = LF_image.shape
    if ij is None:
        LF = LF_image.transpose(0, 2, 1, 3, 4).reshape(N * M, N * M, 3)
    else:
        i, j = ij
        LF = LF_image[i][j]
    im = np.array(LF)
    if resize_to is not None:
        resize_to = max(LF.shape[2], resize_to)
        if ij is None:
            resize_to *= N
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


if __name__ == "__main__":
    pass
