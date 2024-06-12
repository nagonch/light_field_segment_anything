from data import HCIOldDataset
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat
from collections import defaultdict
from tqdm import tqdm
from utils import visualize_segmentation_mask


def get_set():
    return set()


def labels_per_pixel(labels, disparity):
    s_size, t_size, u_size, v_size = labels.shape
    u_space, v_space = torch.meshgrid(
        (torch.arange(u_size).cuda(), torch.arange(v_size).cuda())
    )
    labels_projected = torch.zeros_like(labels).cuda()
    for s in range(s_size):
        for t in range(t_size):
            disp = disparity[s, t]
            l = labels[s, t]
            u_st = (u_space + disp[:, :, 0]).reshape(-1).long()
            v_st = (v_space + disp[:, :, 1]).reshape(-1).long()
            mask = (u_st >= 0) & (u_st < u_size) & (v_st >= 0) & (v_st < v_size)
            labels_projected[s, t, u_st[mask], v_st[mask]] = l[
                u_space.reshape(-1)[mask], v_space.reshape(-1)[mask]
            ]
    lengths = []
    for label_set in labels_projected.reshape(s_size * t_size, u_size * v_size).T:
        lengths.append(len(set(label_set.tolist())))
    result = torch.tensor(lengths).cuda().float().mean()
    return result, labels_projected


if __name__ == "__main__":
    dataset = HCIOldDataset()
    LF = dataset.get_scene("papillon")
    disparity = torch.tensor(dataset.get_disparity("papillon")).cuda()
    labels = torch.tensor(torch.load("past_merges/merged_papillon.pt")).cuda()
    result, labels_projected = labels_per_pixel(labels, disparity)
    visualize_segmentation_mask(labels.detach().cpu().numpy(), None)
    visualize_segmentation_mask(labels_projected.detach().cpu().numpy(), None)
    print(result)
