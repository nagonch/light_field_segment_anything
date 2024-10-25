from data2 import UrbanLFSynDataset
from utils import visualize_segmentation_mask
from plenpy.lightfields import LightField
import numpy as np
import torch


class ConsistencyMetrics:
    def __init__(self, predicted_labels, gt_disparity):
        predictions = torch.tensor(predicted_labels).cuda()  # [n, s, t, u, v]
        disparity = torch.tensor(gt_disparity.copy()).cuda()
        self.labels_projected = torch.zeros_like(predictions).cuda()
        for i, prediction in enumerate(predictions):
            print(prediction.shape)


if __name__ == "__main__":
    dataset = UrbanLFSynDataset(
        "/home/nagonch/repos/LF_object_tracking/UrbanLF_Syn/val"
    )
    for LF, labels, disparity in dataset:
        labels = np.stack([labels == i for i in np.unique(labels)])
        metrics = ConsistencyMetrics(labels, disparity)
        # visualize_segmentation_mask(labels)
