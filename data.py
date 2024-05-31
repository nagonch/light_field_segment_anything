import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import math
import h5py


class UrbanLFDataset(Dataset):
    def __init__(self, section, return_disparity=False):
        self.data_path = f"UrbanLF_Syn/{section}"
        self.return_disparity = return_disparity
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
        if self.return_disparity and disparities:
            disparities = np.stack(disparities).reshape(
                n_apertures,
                n_apertures,
                u,
                v,
            )
            return LF, disparities
        else:
            return LF


class HCIOldDataset:
    def __init__(self, data_path="HCI_dataset_old"):
        self.data_path = data_path
        self.scene_to_path = {}
        for scene in [
            "buddha2",
            "horses",
            "papillon",
            "stillLife",
        ]:
            self.scene_to_path[scene] = f"{data_path}/{scene}""

    def get_scene(self, name):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        LF = torch.tensor(np.array(scene["LF"])).cuda()
        return LF


if __name__ == "__main__":
    from plenpy.lightfields import LightField

    HCI_dataset = HCIOldDataset()
    LF = LightField(HCI_dataset.get_scene("buddha2").detach().cpu().numpy())
    LF.show()
