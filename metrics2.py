from data2 import UrbanLFSynDataset, HCIOldDataset
from utils import visualize_segmentation_mask
from plenpy.lightfields import LightField
import numpy as np
import torch
from LF_image_sam_seg import masks_to_segments
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ConsistencyMetrics:
    def __init__(self, predicted_masks, gt_disparity):
        s_size, t_size = gt_disparity.shape[:2]
        predictions = predicted_masks  # [n, s, t, u, v]
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
                    coords = torch.nonzero(mask[s, t])
                    centroid = coords.float().mean(axis=0)
                    values_i.append(torch.norm(centroid - centroid_orig))
            values_i = torch.stack(values_i)
            values_i = values_i[~torch.isnan(values_i)]
            if values_i.shape[0] > 0:
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


class AccuracyMetrics:
    def __init__(self, predicted_segments, gt_segments, only_central_subview=False):
        if only_central_subview:
            s, t, u, v = predicted_segments.shape
            predicpredicted_segmentstions = predicted_segments[s // 2, t // 2, :, :][
                None, None, :, :
            ]
            gt_segments = gt_segments[None, None, :, :]
        self.predictions = predicted_segments
        self.gt_labels = torch.tensor(gt_segments.copy()).cuda()
        self.s, self.t, self.u, self.v = self.predictions.shape
        self.n_pixels = self.s * self.t * self.u * self.v
        self.boundary_d = 2

    def achievable_accuracy(self):
        """
        M. Y. Lui, O. Tuzel, S. Ramalingam, R. Chellappa.
        Entropy rate superpixel segmentation.
        IEEE Conference on Computer Vision and Pattern Recognition, 2011, pp. 2097-2104.
        """
        predictions_modified = torch.zeros_like(self.predictions).long()
        for label in torch.unique(self.predictions)[1:]:
            mask = self.predictions == label
            gt_label = self.gt_labels[mask]
            predictions_modified[self.predictions == label] = torch.mode(
                gt_label  # superpixel's GT label is the GT label it intersects with highest area
            ).values.long()
        predictions_modified_reshape = predictions_modified.reshape(-1)
        result = (
            (
                predictions_modified_reshape[predictions_modified_reshape != 0]
                == self.gt_labels.reshape(-1)[predictions_modified_reshape != 0]
            )
            .float()
            .mean()
            .item()
        )  # label 0 is considered "unsegmented region" outside of the coverage for our method
        return result, predictions_modified

    def boundary_recall(self):
        """
        P. Neubert, P. Protzel.
        Superpixel benchmark and comparison.
        Forum Bildverarbeitung, 2012.
        """
        true_positives = 0
        totals = 0
        visualization = torch.zeros_like(self.predictions)
        for s in range(self.gt_labels.shape[0]):
            for t in range(self.gt_labels.shape[1]):
                gradient_x_gt, gradient_y_gt = torch.gradient(
                    self.gt_labels[s, t].float()
                )
                edges_gt = (torch.sqrt(gradient_x_gt**2 + gradient_y_gt**2) > 0).long()
                gradient_x, gradient_y = torch.gradient(self.predictions[s, t].float())
                edges_pred = (torch.sqrt(gradient_x**2 + gradient_y**2) > 0).long()
                kernel = torch.ones((1, 1, 5, 5)).cuda()
                d_map = F.conv2d(
                    F.pad(edges_pred, (2, 2, 2, 2), value=0)[None][None].float(),
                    kernel.float(),
                )
                d_map = (d_map > 0).long()[0, 0]
                result_values = d_map[edges_gt == 1]
                visualization[s, t] = 2 * d_map + (edges_gt == 1).long()
                true_positives += result_values.sum().item()
                totals += result_values.shape[0]
        return true_positives / totals, visualization

    def coverage(self):
        return (self.predictions >= 1).float().mean().item()

    def size_metrics(self, eps=1e-9):
        s, t, u, v = self.predictions.shape
        superpixel_sizes = []
        gt_segment_sizes = []
        for label in torch.unique(self.predictions)[1:]:
            mask = self.predictions == label
            gt_labels = self.gt_labels[mask]
            gt_label = torch.mode(gt_labels).values.long()
            gt_mask = self.gt_labels == gt_label
            mask_size = mask.sum() / (s * t)
            gt_mask_size = gt_mask.sum() / (s * t)
            superpixel_sizes.append(mask_size)
            gt_segment_sizes.append(gt_mask_size)
        superpixel_sizes = torch.tensor(superpixel_sizes).cuda().float()
        gt_segment_sizes = torch.tensor(gt_segment_sizes).cuda().float()
        mean_superpixel_size = superpixel_sizes.mean().item()
        superpixel_to_gt_segment_ratio = (
            (superpixel_sizes / (gt_segment_sizes + eps)).mean().item()
        )
        return mean_superpixel_size, superpixel_to_gt_segment_ratio

    def compactness(self, eps=1e-9):
        """
        A. Schick, M. Fischer, R. Stiefelhagen.
        Measuring and evaluating the compactness of superpixels.
        International Conference on Pattern Recognition, 2012, pp. 930-934.
        """
        result = 0
        for label in torch.unique(self.predictions)[1:]:
            mask = self.predictions == label
            s, t, u, v = mask.shape
            for s_i in range(s):
                for t_i in range(t):
                    area = mask[s_i, t_i].sum()
                    gradient_x, gradient_y = torch.gradient(mask[s_i, t_i].float())
                    edges = (torch.sqrt(gradient_x**2 + gradient_y**2) > 0).long()
                    perim = edges.sum()
                    Q_s = 4 * torch.pi * area / (perim**2 + eps)
                    result += Q_s * area / (u * v)
        result = result / (s * t)
        return result.item()

    def undersegmentation_error(self):
        """
        P. Neubert, P. Protzel.
        Superpixel benchmark and comparison.
        Forum Bildverarbeitung, 2012.
        """
        undersegmentation_errors = []
        for label in torch.unique(self.gt_labels):
            total_penalty = 0
            gt_region = self.gt_labels == label
            superpixel_labels = torch.unique(self.predictions[gt_region])
            for superpixel_label in superpixel_labels[1:]:
                predicted_region = self.predictions == superpixel_label
                overlap = (predicted_region.long() * gt_region.long()).sum()
                total_penalty += min(overlap, predicted_region.sum() - overlap)
            undersegmentation_errors.append(total_penalty / gt_region.sum())
        return torch.tensor(undersegmentation_errors).cuda().mean().item()

    def get_metrics_dict(self):
        achievable_accuracy, _ = self.achievable_accuracy()
        boundary_recall, _ = self.boundary_recall()
        coverage = self.coverage()
        mean_superpixel_size, superpixel_to_gt_segment_ratio = self.size_metrics()
        compactness = self.compactness()
        undersegmentation_error = self.undersegmentation_error()
        result = {
            "achievable_accuracy": achievable_accuracy,
            "boundary_recall": boundary_recall,
            "coverage": coverage,
            "mean_superpixel_size": mean_superpixel_size,
            "superpixel_to_gt_segment_ratio": superpixel_to_gt_segment_ratio,
            "compactness": compactness,
            "undersegmentation_error": undersegmentation_error,
        }
        return result


if __name__ == "__main__":
    pass
