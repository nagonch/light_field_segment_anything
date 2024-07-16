from data import HCIOldDataset, UrbanLFDataset, MMSPG
from utils import EXP_CONFIG, visualize_segmentation_mask
import torch
import os

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

exp_dir = f"experiments/{EXP_CONFIG['exp-name']}"
for i in range(len(dataset)):
    if not os.path.exists(f"{exp_dir}/{str(i).zfill(4)}_result.pth"):
        continue
    LF, _, _ = dataset[i]
    mask = torch.load(f"{exp_dir}/{str(i).zfill(4)}_result.pth").cpu().numpy()
    visualize_segmentation_mask(mask, LF)
    visualize_segmentation_mask(mask, LF, only_boundaries=True)
