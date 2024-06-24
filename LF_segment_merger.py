import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    get_subview_indices,
    MERGER_CONFIG,
    get_process_to_segments_dict,
    resize_LF,
)
import os
import numpy as np
from PIL import Image
from optimizer import GreedyOptimizer

# import torch.multiprocessing as mp
import multiprocessing as mp

mp.set_start_method("spawn", force=True)


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
        if self.verbose:
            os.makedirs("LF_ransac_output", exist_ok=True)

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
        if self.verbose:
            os.makedirs(
                f"LF_ransac_output/{central_mask_num.item()}/fit/{str(s.item()).zfill(2)}_{str(t.item()).zfill(2)}",
                exist_ok=True,
            )
            segments_ordered = subview_segments[
                torch.argsort(similarities, descending=True)
            ][: MERGER_CONFIG["top-n-segments-visualization"]]
            for top_i, number in enumerate(segments_ordered):
                image = self.get_segment_image(number, s, t)
                image.save(
                    f"LF_ransac_output/{central_mask_num.item()}/fit/{str(s.item()).zfill(2)}_{str(t.item()).zfill(2)}/{str(top_i).zfill(2)}.png"
                )
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
        if self.verbose:
            os.makedirs(f"LF_ransac_output/{central_mask_num.item()}", exist_ok=True)
            im = self.get_segment_image(
                central_mask_num, self.s_central, self.t_central
            )
            im.save(f"LF_ransac_output/{central_mask_num}/main.png")
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
        for segment_num in self.central_segments:
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


def parallelize_segments(i, results, segments, proc_to_seg_dict, embeddings):
    segments_i = (
        segments.float() * torch.isin(segments, proc_to_seg_dict[i]).float()
    ).long()
    merger = LF_segment_merger(segments_i, embeddings)
    results[i] = merger.get_result_masks().cpu()


def get_merged_segments(segments, embeddings):
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    if (
        torch.unique(segments[s_central, t_central]).shape[0]
        >= MERGER_CONFIG["min-central-segments-for-parallel"]
    ):
        proc_to_seg_dict = get_process_to_segments_dict(embeddings)
        result_segments_list = mp.Manager().list(
            [None] * MERGER_CONFIG["n-parallel-processes"]
        )
        processes = []
        for rank in range(MERGER_CONFIG["n-parallel-processes"]):
            p = mp.Process(
                target=parallelize_segments,
                args=(
                    rank,
                    result_segments_list,
                    segments,
                    proc_to_seg_dict,
                    embeddings,
                ),
            )
            p.start()
            processes.append(p)
        # Wait for all processes to complete
        for p in tqdm(processes):
            p.join()
        result = torch.stack(list(result_segments_list)).sum(axis=0)
    else:
        merger = LF_segment_merger(segments, embeddings)
        result = merger.get_result_masks()
    return result


if __name__ == "__main__":
    pass
    # from scipy.io import loadmat
    # from data import UrbanLFDataset

    # segments = torch.load("segments.pt").cuda()
    # embeddings = torch.load("embeddings.pt")
    # dataset = UrbanLFDataset("val")
    # LF = dataset[3].detach().cpu().numpy()
    # merger = LF_segment_merger(segments, embeddings, LF)
    # result_masks = merger.get_result_masks()
    # torch.save(result_masks, "merged.pt")
    # print(result_masks)
