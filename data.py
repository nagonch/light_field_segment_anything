import numpy as np
from PIL import Image
import os
import math
import h5py
from plenpy.lightfields import LightField
from utils import visualize_segmentation_mask


class HCIOldDataset:
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
        LF = np.flip(LF, axis=0)
        return LF

    def get_labels(self, name):
        labels = np.array(
            h5py.File(f"{self.scene_to_path[name]}/labels.h5", "r")["GT_LABELS"]
        )
        labels = np.flip(labels, axis=0)
        return labels

    def get_disparity(self, name, eps=1e-9):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        gt_depth = np.array(scene["GT_DEPTH"])
        s_size, t_size = gt_depth.shape[:2]
        gt_disparity = np.zeros_like(gt_depth)
        dH = scene.attrs["dH"][0]
        f = scene.attrs["focalLength"][0]
        shift = scene.attrs["shift"][0]
        for s in range(s_size):
            for t in range(t_size):
                gt_disparity[s, t, :, :] = dH * f / (gt_depth[s, t] + eps) - shift
        return gt_disparity

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        LF = self.get_scene(scene_name)
        labels = self.get_labels(scene_name)
        disparity = self.get_disparity(scene_name)
        return LF, labels, disparity


class UrbanLFSynDataset:
    def __init__(self, data_path="UrbanLF_Syn/val"):
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
                img = np.array(Image.open(f"{self.data_path}/{frame}/{filename}"))[
                    :, :, :3
                ]
                imgs.append(img)
            elif filename.endswith("disparity.npy"):
                disparities.append(np.load(f"{self.data_path}/{frame}/{filename}"))
            elif filename.endswith("label.npy"):
                labels.append(np.load(f"{self.data_path}/{frame}/{filename}"))
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
        disparities = np.stack(disparities).reshape(
            n_apertures,
            n_apertures,
            u,
            v,
        )
        disparities = np.flip(disparities, axis=(0, 1))
        return LF, labels, disparities


class UrbanLFRealDataset:
    def __init__(self, data_path="UrbanLF_Real/val"):
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
        for filename in sorted(os.listdir(f"{self.data_path}/{frame}")):
            if filename == "label.png":
                continue
            elif filename == "label.npy":
                label = np.load(f"{self.data_path}/{frame}/label.npy")
            else:
                img = np.array(Image.open(f"{self.data_path}/{frame}/{filename}"))[
                    :, :, :3
                ]
                imgs.append(img)
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
        return LF, label


class MMSPG:
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
        LF = LF[3:-3, 3:-3]  # drop subviews affected by vignetting
        LF = (LF // 256).astype(np.uint8)
        LF = np.flip(LF, axis=(0, 1))
        return LF, None, None


if __name__ == "__main__":
    pass
