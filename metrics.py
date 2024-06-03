from data import HCIOldDataset
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat
from collections import defaultdict
from tqdm import tqdm


def get_set():
    return set()


def labels_per_pixel(labels, disparity):
    s_size, t_size, u_size, v_size = labels.shape
    m = defaultdict(get_set)
    for s in tqdm(range(s_size)):
        for t in tqdm(range(t_size)):
            for u in tqdm(range(u_size)):
                for v in range(v_size):
                    label = labels[s, t, u, v]
                    u_center = (u + disparity[s, t, u, v, 0]).long()
                    v_center = (v + disparity[s, t, u, v, 1]).long()
                    if (
                        u_center >= 0
                        and v_center >= 0
                        and u_center < u_size
                        and v_center < v_size
                    ):
                        m[f"{str(u_center).zfill(4)}_{str(v_center).zfill(4)}"].add(
                            label.item()
                        )
    pixel_lengths = []
    for list in m.values():
        pixel_lengths.append(len(list))
    metric = torch.tensor(pixel_lengths).cuda().float().mean()
    return metric.item()


if __name__ == "__main__":
    dataset = HCIOldDataset()
    LF = dataset.get_scene("papillon")
    disparity = torch.tensor(dataset.get_disparity("papillon")).cuda()
    labels = torch.tensor(torch.load("merged.pt")).cuda()
    result = labels_per_pixel(labels, disparity)
    print(result)
