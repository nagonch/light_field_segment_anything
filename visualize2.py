from data import HCIOldDataset, UrbanLFDataset, MMSPG
from utils import EXP_CONFIG, visualize_segmentation_mask
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

name_to_dataset = {
    "HCI": HCIOldDataset(),
    "URBAN_SYN": UrbanLFDataset("UrbanLF_Syn/val"),
    "URBAN_REAL": UrbanLFDataset("UrbanLF_Real/val"),
    "MMSPG": MMSPG(),
}
datset_name = EXP_CONFIG["dataset-name"]
dataset = name_to_dataset.get(datset_name)
if not dataset:
    raise ValueError(f"{EXP_CONFIG['dataset-name']} is not a valid datset name")


def filter_the_mask(segments, min_mask_area=100, subview_percentage=0.9):
    s, t, u, v = segments.shape
    min_mask_area = min_mask_area
    for i in torch.unique(segments)[1:]:
        seg_i = (segments == i).float()
        seg_sum = seg_i.sum(axis=(2, 3))
        if not (
            seg_sum.mean() >= min_mask_area
            and (seg_sum > min_mask_area).sum() / (s * t) >= subview_percentage
        ):
            segments[segments == i] = 0
    return segments


def save_LF(LF, folder):
    s, t, u, v, c = LF.shape
    for i in range(s):
        iterator = range(t) if i % 2 == 0 else range(t - 1, -1, -1)
        for j, j_real in zip(iterator, range(t)):
            img = Image.fromarray(LF[i][j])
            img.save(f"{folder}/{str(i).zfill(4)}_{str(j_real).zfill(4)}.png")


LF_ONLY = False

exp_dir = f"experiments/{EXP_CONFIG['exp-name']}"
for i in tqdm(range(len(dataset))):
    if not os.path.exists(f"{exp_dir}/{str(i).zfill(4)}_result.pth"):
        continue
    LF, _, _ = dataset[i]
    os.makedirs(f"{exp_dir}/{str(i).zfill(4)}", exist_ok=True)
    if LF_ONLY:
        save_LF(LF, f"{exp_dir}/{str(i).zfill(4)}")
    else:
        os.makedirs(f"{exp_dir}/{str(i).zfill(4)}_mask", exist_ok=True)
        os.makedirs(f"{exp_dir}/{str(i).zfill(4)}_bound", exist_ok=True)
        mask = torch.load(f"{exp_dir}/{str(i).zfill(4)}_result.pth")
        mask = filter_the_mask(mask)
        mask = mask.cpu().numpy()
        vis_mask = visualize_segmentation_mask(mask, LF, just_return=True)
        vis_boundaries = (
            visualize_segmentation_mask(
                mask, LF, only_boundaries=True, just_return=True
            )
            * 255
        ).astype(np.uint8)
        save_LF(vis_mask, f"{exp_dir}/{str(i).zfill(4)}_mask")
        save_LF(vis_boundaries, f"{exp_dir}/{str(i).zfill(4)}_bound")
