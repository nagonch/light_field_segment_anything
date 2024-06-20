from data import HCIOldDataset, UrbanLFDataset
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat
from collections import defaultdict
from tqdm import tqdm
from utils import visualize_segmentation_mask
import torch.nn.functional as F


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

    def self_similarity(self, eps=1e-9):
        """
        Zhu, Hao, Qi Zhang, and Qing Wang.
        "4D light field superpixel and segmentation."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
        """
        labels_projected = self.labels_projected
        results = []
        for label_number in torch.unique(
            labels_projected[
                labels_projected.shape[0] // 2, labels_projected.shape[1] // 2
            ]
        )[1:]:
            mask_filtered = (self.labels_projected == label_number).to(torch.int32)
            masks = mask_filtered.reshape(
                -1, mask_filtered.shape[-2], mask_filtered.shape[-1]
            )
            masks_x, masks_y = torch.meshgrid(
                (
                    torch.arange(masks.shape[1]).cuda(),
                    torch.arange(masks.shape[2]).cuda(),
                ),
                indexing="ij",
            )
            masks_x = masks_x.repeat(masks.shape[0], 1, 1)
            masks_y = masks_y.repeat(masks.shape[0], 1, 1)
            centroids_x = (masks_x * masks).sum(axis=(1, 2)) / (
                masks.sum(axis=(1, 2)) + eps
            )
            centroids_y = (masks_y * masks).sum(axis=(1, 2)) / (
                masks.sum(axis=(1, 2)) + eps
            )
            centroids = torch.stack((centroids_y, centroids_x)).T
            main_centroid = centroids[centroids.shape[0] // 2]
            metric = torch.norm(centroids - main_centroid, p=2, dim=1)
            metric = metric.sum() / (metric.shape[0] - 1)
            results.append(metric)
        return torch.tensor(metric).mean()


class AccuracyMetrics:
    def __init__(self, predictions, gt_labels):
        self.predictions = predictions
        self.gt_labels = gt_labels
        s, t, u, v = self.predictions.shape
        self.n_pixels = s * t * u * v
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
                gt_label
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
        )
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
        return (self.predictions > 0).float().mean().item()

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


if __name__ == "__main__":
    dataset = UrbanLFDataset("val", return_labels=True)
    LF, labels = dataset[3]
    labels = labels[2:-2, 2:-2]
    predictions = torch.tensor(torch.load("merged.pt")).cuda()
    acc_metrics = AccuracyMetrics(predictions, labels)
    recall, visualization = acc_metrics.boundary_recall()
    print(recall)
    # print(acc_metrics.undersegmentation_error())
    # achievable_accuracy, predictions_modified = acc_metrics.achievable_accuracy()
    # print(achievable_accuracy)
    # # coverage = acc_metrics.coverage()
    visualize_segmentation_mask(labels.cpu().numpy(), None)
    visualize_segmentation_mask(predictions.cpu().numpy(), None)
    visualize_segmentation_mask(visualization.cpu().numpy(), None)
    # visualize_segmentation_mask(predictions_modified.cpu().numpy(), None)
