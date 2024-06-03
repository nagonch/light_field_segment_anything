from data import HCIOldDataset
import torch


def labels_per_pixel():
    pass


if __name__ == "__main__":
    dataset = HCIOldDataset()
    LF = dataset.get_scene("papillon")
    disparity = dataset.get_disparity("papillon")
    masks = torch.load("merged.pt")
    print(LF.shape, disparity.shape, masks.shape)
