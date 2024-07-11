import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import math
import h5py
from torch.nn.functional import normalize
from scipy.io import loadmat

# from utils import remap_labels


class UrbanLFDataset(Dataset):
    def __init__(self, data_path, return_disparity=True, return_labels=True):
        self.data_path = data_path
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
                disparities.append(
                    torch.tensor(np.load(f"{self.data_path}/{frame}/{filename}"))
                )
            elif filename.endswith("label.npy"):
                labels.append(
                    torch.tensor(np.load(f"{self.data_path}/{frame}/{filename}"))
                )
        LF = torch.stack(imgs)
        n_apertures = int(math.sqrt(LF.shape[0]))
        u, v, c = LF.shape[-3:]
        LF = LF.reshape(
            n_apertures,
            n_apertures,
            u,
            v,
            c,
        ).numpy()
        return_tuple = [
            LF,
        ]
        if self.return_labels and labels:
            if len(labels) == 1:
                return_tuple.append(labels[0])
            else:
                labels = (
                    torch.stack(labels)
                    .reshape(
                        n_apertures,
                        n_apertures,
                        u,
                        v,
                    )
                    .numpy()
                )
                # labels = remap_labels(labels)
                labels += 1
                return_tuple.append(labels)
        elif self.return_labels:
            return_tuple.append(None)
        if self.return_disparity and disparities:
            disparities = (
                torch.stack(disparities)
                .reshape(
                    n_apertures,
                    n_apertures,
                    u,
                    v,
                )
                .cuda()
            )
            s_size, t_size, u_size, v_size = disparities.shape
            disparities_result = torch.zeros(
                (
                    n_apertures,
                    n_apertures,
                    u,
                    v,
                    2,
                )
            ).cuda()
            for s in range(s_size):
                for t in range(t_size):
                    baseline = (
                        torch.tensor([s_size // 2 - s, t_size // 2 + t]).float().cuda()
                    )
                    disparities_result[s, t] = disparities[s, t][:, :, None] * baseline
            return_tuple.append(disparities_result)
        elif self.return_disparity:
            return_tuple.append(None)
        return return_tuple


class HCIOldDataset(Dataset):
    def __init__(self, data_path="HCI_dataset_old"):
        self.data_path = data_path
        self.scene_to_path = {}
        self.scenes = [
            "horses",
            "papillon",
            "stillLife",
            "buddha",
        ]
        for scene in self.scenes:
            self.scene_to_path[scene] = f"{data_path}/{scene}"

    def get_scene(self, name):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        LF = np.array(scene["LF"])
        return LF

    def get_labels(self, name):
        labels = np.array(
            h5py.File(f"{self.scene_to_path[name]}/labels.h5", "r")["GT_LABELS"]
        )
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

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        LF = self.get_scene(scene_name)
        labels = self.get_labels(scene_name)
        disparity = self.get_disparity(scene_name)
        return LF, labels, disparity


class MMSPG(Dataset):
    def __init__(self, convert=True):
        self.path = "MMSPG"
        self.scenes = os.listdir(self.path)
        self.convert = convert

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = f"{self.path}/{self.scenes[idx]}"
        LF = h5py.File(scene_path, "r")["LF"]
        LF = np.transpose(LF, (4, 3, 2, 1, 0))[:, :, :, :, :3]
        LF = LF[3:-3, 3:-3]
        if self.convert:
            LF = LF.astype(np.int32)
        return LF, None, None


if __name__ == "__main__":
    pass
