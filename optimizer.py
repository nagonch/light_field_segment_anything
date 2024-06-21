import torch
from utils import unravel_index, MERGER_CONFIG
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F


class GreedyOptimizer:
    def __init__(
        self,
        similarities,
        segment_matrix,
        central_segment,
        segment_indices,
        central_segment_num,
        LF,
        verbose=MERGER_CONFIG["verbose"],
        lambda_reg=MERGER_CONFIG["lambda-reg"],
        min_similarity=MERGER_CONFIG["min-similarity"],
    ):
        self.central_segment_num = central_segment_num
        self.LF = LF
        self.verbose = verbose
        self.segment_indices = segment_indices
        self.min_similarity = min_similarity
        self.segment_matrix = segment_matrix
        self.central_segment = central_segment
        self.lambda_reg = lambda_reg
        self.n_subviews, self.n_segments = similarities.shape
        self.similarities = similarities
        self.similarities_initial = torch.clone(self.similarities)
        self.mask_centroids, self.central_segment_centroid = self.get_masks_centroids()
        self.reg_matrix = torch.zeros_like(similarities).cuda()
        if self.verbose:
            os.makedirs(
                f"LF_ransac_output/optimizer/{central_segment_num}", exist_ok=True
            )
            central_segment_im = self.get_segment_image(
                self.central_segment, self.LF.shape[0] // 2, self.LF.shape[1] // 2
            )
            central_segment_im.save(
                f"LF_ransac_output/optimizer/{central_segment_num}/central.png"
            )

    @torch.no_grad()
    def get_segment_image(self, segment, s, t):
        mask_image = (
            (self.LF[s, t] * segment[:, :, None])
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        mask_image[mask_image == 0] = 255
        im = Image.fromarray(mask_image)
        return im

    def get_masks_centroids(self, eps=1e-9):
        masks = self.segment_matrix.reshape(
            -1, self.segment_matrix.shape[-2], self.segment_matrix.shape[-1]
        )
        masks = torch.cat((masks, self.central_segment[None]), dim=0)
        masks_x, masks_y = torch.meshgrid(
            (torch.arange(masks.shape[1]).cuda(), torch.arange(masks.shape[2]).cuda()),
            indexing="ij",
        )
        masks_x = masks_x.repeat(masks.shape[0], 1, 1)
        centroids_x = (masks_x * masks).sum(axis=(1, 2)) / (
            masks.sum(axis=(1, 2)) + eps
        )
        del masks_x
        masks_y = masks_y.repeat(masks.shape[0], 1, 1)
        centroids_y = (masks_y * masks).sum(axis=(1, 2)) / (
            masks.sum(axis=(1, 2)) + eps
        )
        del masks_y
        centroids = torch.stack((centroids_y, centroids_x)).T
        centroids -= centroids.mean(axis=0)
        return (
            centroids[:-1].reshape(
                self.n_subviews,
                self.n_segments,
                2,
            ),
            centroids[-1],
        )

    def svd_regularization(self, segment_inds, eps=1e-9):
        centroids = self.mask_centroids[segment_inds[:, 0], segment_inds[:, 1]]
        centroids = torch.cat((centroids, self.central_segment_centroid[None]), dim=0)
        cov = torch.cov(centroids.T)
        sing_values = torch.svd(cov, compute_uv=False)[1]
        score = sing_values.min() / (sing_values.max() + eps)

        return score

    def update_reg(self, subview_ind, segment_ind, chosen_segment_inds=None):
        if chosen_segment_inds:
            reg_segment_ids = chosen_segment_inds + [
                torch.tensor([subview_ind, segment_ind]),
            ]
            reg_segment_ids = torch.stack(reg_segment_ids).cuda()
            self.reg_matrix[subview_ind, segment_ind] = self.svd_regularization(
                reg_segment_ids
            )

    def run(self):
        matches = []
        chosen_segment_inds = []
        for i in range(self.n_subviews):
            function_val = (
                1 - self.lambda_reg
            ) * self.similarities + self.lambda_reg * self.reg_matrix
            ind_num, segment_num = unravel_index(
                function_val.argmax(), (self.n_subviews, self.n_segments)
            )
            if self.verbose:
                os.makedirs(
                    f"LF_ransac_output/optimizer/{self.central_segment_num}/{str(i).zfill(4)}",
                    exist_ok=True,
                )
                function_squeezed = function_val.reshape(-1)
                segments_squeezed = self.segment_matrix.reshape(
                    -1, self.segment_matrix.shape[-2], self.segment_matrix.shape[-1]
                )
                sorted = torch.argsort(function_squeezed, descending=True)[
                    : MERGER_CONFIG["top-n-segments-visualization"]
                ]
                segments_squeezed = segments_squeezed[sorted, :, :]
                sorted = unravel_index(sorted, (self.n_subviews, self.n_segments))
                for top_k, (segment_vis, subview_ind) in enumerate(
                    zip(segments_squeezed, sorted[:, 0])
                ):
                    s, t = unravel_index(
                        subview_ind, (self.LF.shape[0], self.LF.shape[1])
                    )
                    seg_img = self.get_segment_image(segment_vis, s, t)
                    seg_img.save(
                        f"LF_ransac_output/optimizer/{self.central_segment_num}/{str(i).zfill(4)}/{str(top_k).zfill(2)}.png"
                    )
            self.similarities[ind_num] = torch.ones_like(self.similarities[ind_num]) * (
                -torch.inf
            )
            matches.append(self.segment_indices[ind_num, segment_num].item())
            chosen_segment_inds.append(torch.tensor([ind_num, segment_num]))
            if i < self.n_subviews - 1:
                for candidate_i in range(self.n_subviews):
                    for candidate_j in range(self.n_segments):
                        if self.similarities[candidate_i, candidate_j] == -torch.inf:
                            continue
                        self.update_reg(
                            candidate_i,
                            candidate_j,
                            chosen_segment_inds=chosen_segment_inds,
                        )
        result_matches = []
        for match, match_ind in zip(matches, chosen_segment_inds):
            if (
                self.similarities_initial[match_ind[0], match_ind[1]]
                >= self.min_similarity
            ):
                result_matches.append(match)
        return result_matches


if __name__ == "__main__":
    pass
    # sim_matrix = torch.load("sim_matrix.pt")
    # segment_matrix = torch.load("segment_matrix.pt")
    # segment_indices = torch.load("segment_indices.pt")
    # central_mask = torch.load("central_mask.pt")
    # from data import UrbanLFDataset
    # from utils import resize_LF

    # dataset = UrbanLFDataset("val")
    # LF = torch.tensor(dataset[3]).detach().cpu().numpy()
    # LF = torch.tensor(resize_LF(LF, 256, 341)).cuda()
    # opt = GreedyOptimizer(
    #     sim_matrix,
    #     segment_matrix,
    #     central_mask,
    #     segment_indices,
    #     3350,
    #     LF,
    # )
    # result = opt.run()
    # print(result)
