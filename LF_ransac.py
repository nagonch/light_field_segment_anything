import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    binary_mask_centroid,
    get_subview_indices,
    calculate_outliers,
    MERGER_CONFIG,
    project_point_onto_line,
    shift_binary_mask,
    get_process_to_segments_dict,
)
import os
import numpy as np
from PIL import Image
from torchmetrics.classification import BinaryJaccardIndex

# import torch.multiprocessing as mp
import multiprocessing as mp

mp.set_start_method("spawn", force=True)


class LF_RANSAC_segment_merger:
    @torch.no_grad()
    def __init__(self, segments, embeddings):
        self.segments = segments
        self.embeddings = embeddings
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2
        self.subview_indices = get_subview_indices(self.s_size, self.t_size)
        self.central_segments = self.get_central_segments()
        self.epipolar_line_vectors = self.get_epipolar_line_vectors()
        self.segments_centroids = self.get_segments_centroids()
        self.verbose = MERGER_CONFIG["verbose"]
        self.embedding_coeff = MERGER_CONFIG["embedding-coeff"]
        if self.verbose:
            os.makedirs("LF_ransac_output", exist_ok=True)

    @torch.no_grad()
    def get_segments_centroids(self):
        centroids_dict = {}
        for segment_i in torch.unique(self.segments):
            centroids_dict[segment_i.item()] = binary_mask_centroid(
                self.segments == segment_i
            )[-2:]
        return centroids_dict

    @torch.no_grad()
    def get_epipolar_line_vectors(self):
        epipolar_line_vectors = (
            torch.tensor([self.s_central, self.t_central]).cuda() - self.subview_indices
        ).float()
        aspect_ratio_matrix = (
            torch.diag(torch.tensor([self.v_size, self.u_size])).float().cuda()
        )  # in case the image is non-square
        epipolar_line_vectors = (aspect_ratio_matrix @ epipolar_line_vectors.T).T
        epipolar_line_vectors = F.normalize(epipolar_line_vectors)
        epipolar_line_vectors = epipolar_line_vectors.reshape(
            self.s_size, self.t_size, 2
        )
        return epipolar_line_vectors

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
    def shuffle_indices(self):
        indices_shuffled = self.subview_indices[
            torch.randperm(self.subview_indices.shape[0])
        ]
        indices_shuffled = torch.stack(
            [
                element
                for element in indices_shuffled
                if (
                    element != torch.tensor([self.s_central, self.t_central]).cuda()
                ).any()
            ]
        )
        return indices_shuffled

    @torch.no_grad()
    def filter_segments(self, subview_segments, central_mask_centroid, s, t):
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
    def calculate_peak_iou(
        self,
        central_mask_num,
        mask_subview_num,
        s,
        t,
        metric=BinaryJaccardIndex().cuda(),
    ):
        mask_subview = self.segments[s, t] == mask_subview_num
        mask_central = self.segments[self.s_central, self.t_central] == central_mask_num
        epipolar_line_point = self.segments_centroids[mask_subview_num.item()]
        displacement = project_point_onto_line(
            epipolar_line_point,
            self.epipolar_line_vectors[s, t],
            self.segments_centroids[central_mask_num.item()],
        )
        vec = torch.round(self.epipolar_line_vectors[s, t] * displacement).long()
        mask_new = shift_binary_mask(mask_subview, vec)
        iou = metric(mask_central, mask_new)
        return iou

    @torch.no_grad()
    def fit(self, central_mask_num, central_mask_centroid, s, t):
        subview_segments = torch.unique(self.segments[s, t])[1:]
        subview_segments = self.filter_segments(
            subview_segments, central_mask_centroid, s, t
        )
        if subview_segments is None:
            return -1, torch.nan, torch.nan
        central_embedding = self.embeddings[central_mask_num.item()][0][None]
        embeddings, subview_segments = self.get_segments_embeddings(subview_segments)
        if subview_segments is None:
            return -1, torch.nan, torch.nan
        central_embedding = torch.repeat_interleave(
            central_embedding, embeddings.shape[0], dim=0
        )
        similarities = F.cosine_similarity(embeddings, central_embedding)
        # iou_per_mask = torch.stack(
        #     [
        #         self.calculate_peak_iou(central_mask_num, mask_num, s, t)
        #         for mask_num in subview_segments
        #     ]
        # ).cuda()
        similarities = (
            self.embedding_coeff
            * similarities
            # + (1 - self.embedding_coeff) * iou_per_mask
        )
        order = torch.argsort(similarities)
        result_segment_index = torch.argmax(similarities).item()
        result_segment = subview_segments[result_segment_index]
        result_similarity = similarities[result_segment_index]
        result_centroid = self.segments_centroids[result_segment.item()]
        result_disparity = torch.norm(result_centroid - central_mask_centroid)
        if result_similarity <= MERGER_CONFIG["metric-threshold"]:
            return -1, torch.nan, torch.nan
        return result_segment, result_disparity, result_similarity

    @torch.no_grad()
    def predict(self, central_mask_num, central_mask_centroid, s, t, disparity):
        subview_segments = torch.unique(self.segments[s, t])[1:]
        subview_segments = self.filter_segments(
            subview_segments, central_mask_centroid, s, t
        )
        if subview_segments is None:
            return -1
        central_embedding = self.embeddings[central_mask_num.item()][0][None]
        embeddings, subview_segments = self.get_segments_embeddings(subview_segments)
        if subview_segments is None:
            return -1
        central_embedding = torch.repeat_interleave(
            central_embedding, embeddings.shape[0], dim=0
        )
        embdding_distances = 1 - F.cosine_similarity(embeddings, central_embedding)
        centroids = torch.stack(
            [self.segments_centroids[segment.item()] for segment in subview_segments]
        ).cuda()
        target_point = (
            central_mask_centroid + self.epipolar_line_vectors[s, t] * disparity
        )
        target_point = target_point.repeat(centroids.shape[0], 1)
        distances = torch.norm(centroids - target_point, dim=1)
        distances = distances / distances.max()
        distances = self.embedding_coeff * embdding_distances + (
            1 - self.embedding_coeff
        ) * (distances)
        result_segment_index = torch.argmin(distances).item()
        result_segment = subview_segments[result_segment_index]
        return result_segment

    @torch.no_grad()
    def calculate_outliers(self, central_segment_num, matches):
        if not matches:
            return 0
        embeddings, subview_segments = self.get_segments_embeddings(
            torch.stack(matches).cuda()
        )
        central_embedding = self.embeddings[central_segment_num.item()][0][None]
        central_embedding = torch.repeat_interleave(
            central_embedding, embeddings.shape[0], dim=0
        )
        similarities = F.cosine_similarity(embeddings, central_embedding)
        outliers = calculate_outliers(similarities)
        return outliers

    @torch.no_grad()
    def find_matches(self, central_mask_num):
        if self.verbose:
            os.makedirs(f"LF_ransac_output/{central_mask_num}", exist_ok=True)
            mask_image = (
                (self.segments[self.s_central, self.t_central] == central_mask_num)
                .to(torch.int32)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            ) * 255
            im = Image.fromarray(mask_image)
            im.save(f"LF_ransac_output/{central_mask_num}/main.png")
        indices_shuffled = self.shuffle_indices()
        best_outliers = torch.inf
        best_disparity = torch.nan
        best_match = []
        for iteration in range(
            min(MERGER_CONFIG["ransac-max-iterations"], indices_shuffled.shape[0])
        ):
            matches = []
            central_mask_centroid = self.segments_centroids[central_mask_num.item()]
            # 1. Sample a random s, t
            s_main, t_main = indices_shuffled[iteration]
            # 2. Find a segment match and a depth "the hard way"
            matched_segment, disparity, certainty = self.fit(
                central_mask_num, central_mask_centroid, s_main, t_main
            )
            # 3. For the rest of s and t find match a closest to the depth using centroids
            for s, t in indices_shuffled:
                match, _, _ = self.fit(  # TODO: replace with predict later
                    central_mask_num,
                    central_mask_centroid,
                    s,
                    t,
                    # disparity,
                )
                if match >= 0:
                    matches.append(match)
            # 4. Calculate outliers and repeat procedure
            outliers = self.calculate_outliers(central_mask_num, matches)
            if outliers < best_outliers:
                best_outliers = outliers
                best_match = matches
                best_disparity = disparity
            if (
                outliers / indices_shuffled.shape[0]
                <= MERGER_CONFIG["ransac-max-outliers"]
            ):
                break
        return best_match, best_disparity

    @torch.no_grad()
    def get_result_masks(self):
        self.merged_segments = []
        disparity_map = {}
        for segment_num in tqdm(self.central_segments):
            segment_embedding = self.embeddings.get(segment_num.item(), None)
            if segment_embedding is None:
                continue
            matches, disparity = self.find_matches(segment_num)
            disparity_map[segment_num.item()] = disparity
            self.segments[torch.isin(self.segments, torch.tensor(matches).cuda())] = (
                segment_num
            )
            self.merged_segments.append(segment_num)
        self.segments[
            ~torch.isin(
                self.segments,
                torch.unique(self.segments[self.s_central, self.t_central]),
            )
        ] = 0  # TODO: check what happens with unmatched
        return self.segments


def parallelize_segments(i, results, segments, proc_to_seg_dict, embeddings):
    segments_i = (
        segments.float() * torch.isin(segments, proc_to_seg_dict[i]).float()
    ).long()
    merger = LF_RANSAC_segment_merger(segments_i, embeddings)
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
        merger = LF_RANSAC_segment_merger(segments, embeddings)
        result = merger.get_result_masks()
    return result


if __name__ == "__main__":
    segments = torch.load("segments.pt").cuda()
    embeddings = torch.load("embeddings.pt")
    merger = LF_RANSAC_segment_merger(segments, embeddings)
    result_masks = merger.get_result_masks()
    torch.save(result_masks, "merged.pt")
    print(result_masks)
