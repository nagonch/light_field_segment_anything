import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import math
from utils import save_LF_image
from torchvision.transforms.functional import resize


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
    pass
    # from torch.nn.functional import interpolate
    # from matplotlib import pyplot as plt

    # dataset = LFDataset("UrbanLF_Syn/val", return_disparity=True)
    # img, disp = dataset[3]
    # disp = torch.tensor(disp).reshape(-1, 480, 640).cuda()[None].permute(1, 0, 2, 3)
    # disp = (
    #     interpolate(disp, (256, 341))[:, 0, :, :]
    #     .reshape(9, 9, 256, 341)
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )
    # segments = torch.load("merged.pt")
    # disparity = (disp * (segments == 3350)).mean()
    # print(disparity)
    # plt.imshow(disp[4, 4], cmap="gray")
    # plt.show()
    # plt.close()
    # print(img.shape, disp.shape)
    # print(img.shape)
    # save_LF_image(img, resize_to=None)
