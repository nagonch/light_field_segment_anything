import numpy as np
import imgviz
from PIL import Image
from matplotlib import pyplot as plt
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import yaml

with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def get_sam():
    sam = sam_model_registry["default"](checkpoint=CONFIG["model-path"])
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


def get_LF(dir):
    subviews = []
    for img in list(sorted(os.listdir(dir))):
        path = dir + "/" + img
        subviews.append(np.array(Image.open(path))[:, :, :3])
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
    plt.show()
    if filename:
        plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    pass
