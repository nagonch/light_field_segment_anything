import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import math
from utils import save_LF_image
from torchvision.transforms.functional import resize


def get_LF(dir):
    subviews = []
    for img in list(sorted(os.listdir(dir))):
        path = dir + "/" + img
        subviews.append(np.array(Image.open(path))[:, :, :3])
    n_apertures = int(np.sqrt(len(subviews)))
    u, v, c = subviews[0].shape
    LF = np.stack(subviews).reshape(n_apertures, n_apertures, u, v, c).astype(np.uint8)

    return LF


class LFDataset(Dataset):
    def __init__(self, data_path, return_disparity=False):
        self.data_path = data_path
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
        if self.return_disparity:
            disparities = np.stack(disparities).reshape(
                n_apertures,
                n_apertures,
                u,
                v,
            )
            return LF, disparities
        else:
            return LF


if __name__ == "__main__":
    dataset = LFDataset("UrbanLF_Syn/val", return_disparity=True)
    img, disp = dataset[0]
    print(img.shape, disp.shape)
    # print(img.shape)
    # save_LF_image(img, resize_to=None)
