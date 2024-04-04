import torch
from utils import CONFIG
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import math
from utils import save_LF_image
import cv2
from torchvision.transforms.functional import resize


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


class LFDataset(Dataset):
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

    def resize_img(self, img):
        u, v = img.shape[:2]
        if u < v:
            aspect = v / u
            u = CONFIG["lf-subview-max-size"]
            v = u * aspect
        else:
            aspect = u / v
            v = CONFIG["lf-subview-max-size"]
            u = v * aspect
        img = cv2.resize(
            img,
            (
                int(v),
                int(u),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
        return img

    def __getitem__(self, idx):
        frame = self.frames[idx]
        imgs = []
        for filename in sorted(os.listdir(f"{self.data_path}/{frame}")):
            if not filename.endswith(".png"):
                continue
            img = np.array(Image.open(f"{self.data_path}/{frame}/{filename}"))
            img = self.resize_img(img)
            img = (torch.tensor(img))[:, :, :3]
            imgs.append(img)
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
        return LF


if __name__ == "__main__":
    dataset = LFDataset("UrbanLF_Syn/test")
    img = dataset[0][2:-2, 2:-2]
    # print(img.shape)
    save_LF_image(img, resize_to=None)
