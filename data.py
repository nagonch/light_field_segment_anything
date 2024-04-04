import torch
from utils import CONFIG
import numpy as np
from PIL import Image
import os


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


if __name__ == "__main__":
    pass
