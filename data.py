import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import math
import h5py


class UrbanLFDataset(Dataset):
    def __init__(self, section, return_disparity=False, return_labels=False):
        self.data_path = f"UrbanLF_Syn/{section}"
        self.return_disparity = return_disparity
        self.return_labels = return_labels
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
                disparities.append(np.load(f"{self.data_path}/{frame}/{filename}"))
            elif filename.endswith("label.npy"):
                labels.append(np.load(f"{self.data_path}/{frame}/{filename}"))
        LF = torch.stack(imgs).cuda()
        n_apertures = int(math.sqrt(LF.shape[0]))
        u, v, c = LF.shape[-3:]
        LF = LF.reshape(
            n_apertures,
            n_apertures,
            u,
            v,
            c,
        )
        return_tuple = [
            LF,
        ]
        if self.return_disparity and disparities:
            disparities = np.stack(disparities).reshape(
                n_apertures,
                n_apertures,
                u,
                v,
            )
            return_tuple.append(disparities)
        if self.return_labels and labels:
            labels = np.stack(labels).reshape(
                n_apertures,
                n_apertures,
                u,
                v,
            )
            return_tuple.append(labels)
        return return_tuple


class HCIOldDataset:
    def __init__(self, data_path="HCI_dataset_old"):
        self.data_path = data_path
        self.scene_to_path = {}
        for scene in [
            "horses",
            "papillon",
            "stillLife",
        ]:
            self.scene_to_path[scene] = f"{data_path}/{scene}"

    def get_scene(self, name):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        LF = np.array(scene["LF"])
        return LF

    def get_labels(self, name):
        labels = h5py.File(f"{self.scene_to_path[name]}/labels.h5", "r")["GT_LABELS"]
        return labels

    def get_depth(self, name):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        gt_depth = np.array(scene["GT_DEPTH"])
        return gt_depth

    def get_disparity(self, name, eps=1e-9):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        gt_depth = np.array(scene["GT_DEPTH"])
        s_size, t_size, u_size, v_size = gt_depth.shape
        dH = scene.attrs["dH"][0]
        f = scene.attrs["focalLength"][0]
        shift = scene.attrs["shift"][0]
        disparity = np.zeros((s_size, t_size, u_size, v_size, 2))
        central_ind = s_size // 2
        for s in range(s_size):
            for t in range(t_size):
                disparity[s, t, :, :, 0] = (dH * (central_ind - s)) * f / (
                    gt_depth[s, t] + eps
                ) - shift * (central_ind - s)
                disparity[s, t, :, :, 1] = (dH * (central_ind - t)) * f / (
                    gt_depth[s, t] + eps
                ) - shift * (central_ind - t)
        return disparity


if __name__ == "__main__":
    from plenpy.lightfields import LightField
    import imgviz
    from utils import visualize_segmentation_mask
    from matplotlib import pyplot as plt
    from scipy import ndimage
    from utils import remap_labels

    # HCI_dataset = HCIOldDataset()
    # LF = HCI_dataset.get_scene("stillLife")
    # labels = HCI_dataset.get_labels("stillLife")
    # s, t, u, v, _ = LF.shape
    # LF_vis = LightField(LF)
    # LF_vis.show()
    # dataset = UrbanLFDataset("val", return_labels=True)
    # LF, labels = dataset[1]
    dataset = HCIOldDataset()
    LF = dataset.get_scene("horses")
    labels = torch.tensor(dataset.get_labels("horses")).cuda()
    # visualize_segmentation_mask(labels, LF)
    labels_remapped = remap_labels(labels)
    visualize_segmentation_mask(labels_remapped, LF)
