from data2 import UrbanLFSynDataset
from utils import visualize_segmentation_mask
from plenpy.lightfields import LightField
import numpy as np
import torch
from LF_image_sam_seg import masks_to_segments
import matplotlib.pyplot as plt


class ConsistencyMetrics:
    def __init__(self, predicted_masks, gt_disparity):
        s_size, t_size = gt_disparity.shape[:2]
        predictions = torch.tensor(predicted_masks).cuda()  # [n, s, t, u, v]
        disparity = torch.tensor(gt_disparity.copy()).cuda()
        self.masks_projected = torch.zeros_like(predictions).cuda()
        for i, prediction in enumerate(predictions):
            for s in range(s_size):
                for t in range(t_size):
                    prediction_st = prediction[s, t]
                    st = torch.tensor([s - s_size // 2, t - t_size // 2]).float().cuda()
                    uv_0 = torch.nonzero(prediction_st)
                    disparities_uv = disparity[s, t][prediction_st].reshape(-1)
                    uv = (uv_0 - disparities_uv.unsqueeze(1) * st).long()
                    u = uv[:, 0]
                    v = uv[:, 1]
                    uv = uv[
                        (u >= 0)
                        & (v >= 0)
                        & (u < prediction_st.shape[0])
                        & (v < prediction_st.shape[1])
                    ]
                    mask_projected = torch.zeros_like(prediction_st)
                    mask_projected[uv[:, 0], uv[:, 1]] = 1
                    self.masks_projected[i, s, t] = mask_projected

    def labels_per_pixel(self):
        """
        Khan, N., Zhang, Q., Kasser, L., Stone, H., Kim, M. H., & Tompkin, J. (2019).
        View-consistent 4D light field superpixel segmentation.
        In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 7811-7819).
        """
        n_labels_at_pixel = self.masks_projected.sum(axis=0).float()
        n_labels_at_pixel = n_labels_at_pixel[
            n_labels_at_pixel > 0
        ]  # remove unsegmetned pixels
        return n_labels_at_pixel.mean().item()

    def self_similarity(self):
        """
        Zhu, Hao, Qi Zhang, and Qing Wang.
        "4D light field superpixel and segmentation."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
        """
        values = []
        s_size, t_size = self.masks_projected.shape[1:3]
        for i, mask in enumerate(self.masks_projected):
            centroid_orig = (
                torch.nonzero(mask[s_size // 2, t_size // 2]).float().mean(axis=0)
            )
            values_i = []
            for s in range(s_size):
                for t in range(t_size):
                    if s == s_size // 2 and t == t_size // 2 or mask[s, t].sum() == 0:
                        continue
                    centroid = torch.nonzero(mask[s, t]).float().mean(axis=0)
                    values_i.append(torch.norm(centroid - centroid_orig))
            values_i = torch.stack(values_i)
            values.append(values_i.mean())
        values = torch.stack(values)
        return values.mean().item()

    def get_metrics_dict(self):
        labels_per_pixel = self.labels_per_pixel()
        self_similarity = self.self_similarity()
        result = {
            "labels_per_pixel": labels_per_pixel,
            "self_similarity": self_similarity,
        }
        return result


if __name__ == "__main__":
    dataset = UrbanLFSynDataset(
        "/home/nagonch/repos/LF_object_tracking/UrbanLF_Syn/val"
    )
    for LF, labels, disparity in dataset:
        # visualize_segmentation_mask(labels)
        labels = np.stack([labels == i for i in np.unique(labels)])
        metrics = ConsistencyMetrics(labels, disparity)
        print(metrics.get_metrics_dict())
        # labels_projected = masks_to_segments(metrics.masks_projected)
        # print(metrics.labels_per_pixel())
        # visualize_segmentation_mask(labels_projected.cpu().numpy())
