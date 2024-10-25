import torch
import numpy as np
from PIL import Image
import os
import math
import h5py
from plenpy.lightfields import LightField
from utils import visualize_segmentation_mask


class UrbanLFSynDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.frames = sorted(
            [
                item
                for item in os.listdir(self.data_path)
                if os.path.isdir(f"{self.data_path}/{item}")
            ]
        )
        self.size = len(self.frames)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        frame = self.frames[idx]
        imgs = []
        disparities = []
        labels = []
        for filename in sorted(os.listdir(f"{self.data_path}/{frame}")):
            if (
                filename.endswith("depth.png")
                or filename.endswith("disparity.png")
                or filename.endswith("label.png")
            ):
                continue
            if filename.endswith(".png"):
                img = np.array(Image.open(f"{self.data_path}/{frame}/{filename}"))
                img = (torch.tensor(img))[:, :, :3]
                imgs.append(img)
            elif filename.endswith("disparity.npy"):
                disparities.append(
                    torch.tensor(np.load(f"{self.data_path}/{frame}/{filename}"))
                )
            elif filename.endswith("label.npy"):
                labels.append(
                    torch.tensor(np.load(f"{self.data_path}/{frame}/{filename}"))
                )
        LF = np.stack(imgs)
        n_apertures = int(math.sqrt(LF.shape[0]))
        u, v, c = LF.shape[-3:]
        LF = LF.reshape(
            n_apertures,
            n_apertures,
            u,
            v,
            c,
        )
        LF = np.flip(LF, axis=(0, 1))
        labels = np.stack(labels).reshape(
            n_apertures,
            n_apertures,
            u,
            v,
        )
        labels = np.flip(labels, axis=(0, 1))
        labels += 1
        return LF, labels


if __name__ == "__main__":
    dataset = UrbanLFSynDataset(
        "/home/nagonch/repos/LF_object_tracking/UrbanLF_Syn/val"
    )
    for LF, labels in dataset:
        visualize_segmentation_mask(labels)
