import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple
import os
import numpy as np
from PIL import Image
import yaml
import warnings

warnings.filterwarnings("ignore")
with open("salads_config.yaml") as f:
    MERGER_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def get_subview_indices(s_size, t_size, remove_central=False):
    rows = torch.arange(s_size).unsqueeze(1).repeat(1, t_size).flatten()
    cols = torch.arange(t_size).repeat(s_size)

    indices = torch.stack((rows, cols), dim=-1).cuda()
    if remove_central:
        indices = torch.stack(
            [
                element
                for element in indices
                if (element != torch.tensor([s_size // 2, t_size // 2]).cuda()).any()
            ]
        )
    return indices


def resize_LF(LF, new_u, new_v):
    s, t, u, v, _ = LF.shape
    results = []
    for s_i in range(s):
        for t_i in range(t):
            subview = Image.fromarray(LF[s_i, t_i]).resize((new_v, new_u))
            subview = np.array(subview)
            results.append(subview)
    return np.stack(results).reshape(s, t, new_u, new_v, 3)


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1).cuda()

    return coord


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
            function_val = torch.nan_to_num(function_val, nan=-torch.inf)
            ind_num, segment_num = unravel_index(
                function_val.argmax(), (self.n_subviews, self.n_segments)
            )  # matches for this iteration
            self.similarities[ind_num] = torch.ones_like(self.similarities[ind_num]) * (
                -torch.inf
            )  # so that matches for this iteration are never chose again
            matches.append(self.segment_indices[ind_num, segment_num].item())
            chosen_segment_inds.append(torch.tensor([ind_num, segment_num]))
            if i < self.n_subviews - 1:
                for candidate_i in range(self.n_subviews):
                    for candidate_j in range(self.n_segments):
                        if self.similarities[candidate_i, candidate_j] == -torch.inf:
                            continue  # don't update regularization for already chosen segments to save time
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


class LF_segment_merger:
    @torch.no_grad()
    def __init__(self, segments, embeddings, LF):
        self.segments = segments
        self.LF = torch.tensor(
            resize_LF(LF, segments.shape[-2], segments.shape[-1])
        ).cuda()
        self.embeddings = embeddings
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2
        self.subview_indices = get_subview_indices(self.s_size, self.t_size)
        self.central_segments = self.get_central_segments()
        self.verbose = MERGER_CONFIG["verbose"]
        self.k_cutoff = MERGER_CONFIG["k-cutoff"]

    @torch.no_grad()
    def get_central_segments(self):
        central_segments = torch.unique(self.segments[self.s_central, self.t_central])[
            1:
        ]
        segment_sums = torch.stack(
            [(self.segments == i).sum() for i in central_segments]
        ).cuda()
        central_segments = central_segments[
            torch.argsort(segment_sums, descending=True)
        ]
        return central_segments

    @torch.no_grad()
    def filter_indices(self):
        indices_filtered = self.subview_indices
        indices_filtered = torch.stack(
            [
                element
                for element in indices_filtered
                if (
                    element != torch.tensor([self.s_central, self.t_central]).cuda()
                ).any()
            ]
        )
        return indices_filtered

    @torch.no_grad()
    def filter_segments(self, subview_segments):
        subview_segments = subview_segments[
            ~torch.isin(subview_segments, torch.tensor(self.merged_segments).cuda())
        ]
        subview_segments = [num for num in subview_segments]
        if not subview_segments:
            return None
        subview_segments = torch.stack(subview_segments)
        return subview_segments

    @torch.no_grad()
    def get_segments_embeddings(self, segment_nums):
        segments_embeddings = []
        segment_nums_filtered = []
        for segment in segment_nums:
            embedding = self.embeddings.get(segment.item(), None)
            if embedding is not None:
                segments_embeddings.append(embedding[0])
                segment_nums_filtered.append(segment)
        if not segment_nums_filtered:
            return None, None
        result = torch.stack(segments_embeddings).cuda()
        segment_nums_filtered = torch.stack(segment_nums_filtered).cuda()
        return result, segment_nums_filtered

    @torch.no_grad()
    def get_segment_image(self, segment_num, s, t):
        mask_image = (
            (self.LF[s, t] * (self.segments == segment_num)[s, t, :, :, None])
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        mask_image[mask_image == 0] = 255
        im = Image.fromarray(mask_image)
        return im

    @torch.no_grad()
    def get_subview_masks_similarities(self, central_mask_num, s, t):
        result_sims = torch.zeros((self.k_cutoff,)).cuda()
        result_segments = -1 * torch.ones_like(result_sims).long()
        subview_segments = torch.unique(self.segments[s, t])[1:]
        subview_segments = self.filter_segments(
            subview_segments,
        )
        if subview_segments is None:
            return result_sims, result_segments
        central_embedding = self.embeddings[central_mask_num.item()][0][None]
        embeddings, subview_segments = self.get_segments_embeddings(subview_segments)
        if subview_segments is None:
            return result_sims, result_segments
        central_embedding = torch.repeat_interleave(
            central_embedding, embeddings.shape[0], dim=0
        )
        similarities = F.cosine_similarity(embeddings, central_embedding)
        subview_segments = subview_segments[
            torch.argsort(similarities, descending=True)
        ][: self.k_cutoff]
        similarities = torch.sort(similarities, descending=True)[0][: self.k_cutoff]
        result_sims[: similarities.shape[0]] = similarities
        result_segments[: subview_segments.shape[0]] = subview_segments
        return result_sims, result_segments

    @torch.no_grad()
    def get_similarity_matrix(self, central_mask_num):
        similarity_matrix = torch.zeros(
            (self.subview_indices.shape[0] - 1, self.k_cutoff)
        ).cuda()
        segment_indices = torch.zeros_like(similarity_matrix).long()
        segments = (
            torch.zeros(
                (
                    self.subview_indices.shape[0] - 1,
                    self.k_cutoff,
                    self.u_size,
                    self.v_size,
                )
            )
            .cuda()
            .long()
        )
        for i_ind, (s, t) in enumerate(self.filter_indices()):
            similarities, subview_segment_indices = self.get_subview_masks_similarities(
                central_mask_num, s, t
            )
            similarity_matrix[i_ind] = similarities
            segment_indices[i_ind] = subview_segment_indices
            for i_segment, subview_segment in enumerate(subview_segment_indices):
                segments[i_ind, i_segment, :, :] = (self.segments == subview_segment)[
                    s, t
                ]
        return similarity_matrix, segments, segment_indices

    @torch.no_grad()
    def find_matches_optimize(self, central_mask_num):
        central_mask = (self.segments == central_mask_num)[
            self.s_central, self.t_central
        ].to(torch.int32)
        sim_matrix, segment_matrix, segment_indices = self.get_similarity_matrix(
            central_mask_num
        )
        optimizer = GreedyOptimizer(
            sim_matrix,
            segment_matrix,
            central_mask,
            segment_indices,
            central_mask_num,
            self.LF,
        )
        matches = optimizer.run()
        return matches

    @torch.no_grad()
    def get_result_masks(self):
        self.merged_segments = []
        for segment_num in tqdm(
            self.central_segments,
            desc="central segment merging",
            position=1,
            leave=False,
        ):
            segment_embedding = self.embeddings.get(segment_num.item(), None)
            if segment_embedding is None:
                continue
            matches = self.find_matches_optimize(segment_num)
            self.segments[torch.isin(self.segments, torch.tensor(matches).cuda())] = (
                segment_num
            )
            self.merged_segments.append(segment_num)
        self.segments[
            ~torch.isin(
                self.segments,
                torch.unique(self.segments[self.s_central, self.t_central]),
            )
        ] = 0
        return self.segments
