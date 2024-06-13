from data import HCIOldDataset
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat
from collections import defaultdict
from tqdm import tqdm
from utils import visualize_segmentation_mask


class ConsistencyMetrics:
    def __init__(self, labels, disparity):
        s_size, t_size, u_size, v_size = labels.shape
        u_space, v_space = torch.meshgrid(
            (torch.arange(u_size).cuda(), torch.arange(v_size).cuda())
        )
        self.labels_projected = torch.zeros_like(labels).cuda()
        for s in range(s_size):
            for t in range(t_size):
                disp = disparity[s, t]
                l = labels[s, t]
                u_st = (u_space - disp[:, :, 0]).reshape(-1)
                v_st = (v_space + disp[:, :, 1]).reshape(-1)
                mask = (u_st >= 0) & (u_st < u_size) & (v_st >= 0) & (v_st < v_size)
                self.labels_projected[s, t, u_st[mask].long(), v_st[mask].long()] = l[
                    u_space.reshape(-1)[mask], v_space.reshape(-1)[mask]
                ]

    def labels_per_pixel(self):
        s_size, t_size, u_size, v_size = self.labels_projected.shape
        labels_projected = self.labels_projected.reshape(
            s_size * t_size, u_size * v_size
        ).T
        labels_projected = labels_projected[
            labels_projected[:, s_size * t_size // 2] != 0
        ]
        lengths = []
        for label_set in labels_projected:
            lengths.append(len(set(label_set.tolist())))
        result = torch.tensor(lengths).cuda().float().mean()
        return result


if __name__ == "__main__":
    dataset = HCIOldDataset()
    LF = dataset.get_scene("stillLife")
    disparity = torch.tensor(dataset.get_disparity("stillLife")).cuda()
    labels = torch.tensor(torch.load("past_merges/merged_still_life.pt")).cuda()
    metrics_estimator = ConsistencyMetrics(labels, disparity)
    print(metrics_estimator.labels_per_pixel())
    # visualize_segmentation_mask(labels.detach().cpu().numpy(), None)
    # visualize_segmentation_mask(labels_projected.detach().cpu().numpy(), None)
