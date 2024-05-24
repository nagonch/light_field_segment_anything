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
    resize_LF,
)
import os
import numpy as np
from PIL import Image
from torchmetrics.classification import BinaryJaccardIndex

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
    def get_segment_image(self, segment_num, s, t):
        mask_image = (
            (self.LF[s, t] * (self.segments == segment_num)[s, t, :, :, None])
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        im = Image.fromarray(mask_image)
        return im

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
        similarities = self.embedding_coeff * similarities
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
        if self.verbose:
            os.makedirs(
                f"LF_ransac_output/{central_mask_num.item()}/predict/{str(s.item()).zfill(2)}_{str(t.item()).zfill(2)}",
                exist_ok=True,
            )
            segments_ordered = subview_segments[torch.argsort(distances)][
                : MERGER_CONFIG["top-n-segments-visualization"]
            ]
            for top_i, number in enumerate(segments_ordered):
                image = self.get_segment_image(number, s, t)
                image.save(
                    f"LF_ransac_output/{central_mask_num.item()}/predict/{str(s.item()).zfill(2)}_{str(t.item()).zfill(2)}/{str(top_i).zfill(2)}.png"
                )
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
            os.makedirs(f"LF_ransac_output/{central_mask_num.item()}", exist_ok=True)
            im = self.get_segment_image(
                central_mask_num, self.s_central, self.t_central
            )
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
    def get_subview_masks_similarities(
        self, central_mask_num, central_mask_centroid, s, t, k_cutoff
    ):
        result_sims = torch.zeros((k_cutoff,)).cuda()
        result_segments = -1 * torch.ones_like(result_sims).long()
        subview_segments = torch.unique(self.segments[s, t])[1:]
        subview_segments = self.filter_segments(
            subview_segments, central_mask_centroid, s, t
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
        similarities = self.embedding_coeff * similarities
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
        ][:k_cutoff]
        similarities = torch.sort(similarities, descending=True)[0][:k_cutoff]
        result_sims[: similarities.shape[0]] = similarities
        result_segments[: subview_segments.shape[0]] = subview_segments
        return result_sims, result_segments

    @torch.no_grad()
    def get_similarity_matrix(self, central_mask_num, k_cutoff=20):
        central_mask_centroid = self.segments_centroids[central_mask_num.item()]
        similarity_matrix = torch.zeros(
            (self.subview_indices.shape[0] - 1, k_cutoff)
        ).cuda()
        segment_indices = torch.zeros_like(similarity_matrix).long()
        segments = (
            torch.zeros(
                (self.subview_indices.shape[0] - 1, k_cutoff, self.u_size, self.v_size)
            )
            .cuda()
            .long()
        )
        for i_ind, (s, t) in enumerate(self.shuffle_indices()):
            similarities, subview_segment_indices = self.get_subview_masks_similarities(
                central_mask_num, central_mask_centroid, s, t, k_cutoff
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
        torch.save(sim_matrix, "sim_matrix.pt")
        torch.save(segment_matrix, "segment_matrix.pt")
        torch.save(segment_indices, "segment_indices.pt")
        torch.save(central_mask, "central_mask.pt")
        raise

    @torch.no_grad()
    def get_result_masks(self):
        self.merged_segments = []
        disparity_map = {}
        for segment_num in tqdm(self.central_segments):
            segment_embedding = self.embeddings.get(segment_num.item(), None)
            if segment_embedding is None:
                continue
            matches, disparity = self.find_matches_optimize(segment_num)
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
    from scipy.io import loadmat

    segments = torch.load("segments.pt").cuda()
    embeddings = torch.load("embeddings.pt")
    LF = loadmat("LF.mat")["LF"]
    merger = LF_segment_merger(segments, embeddings, LF)
    result_masks = merger.get_result_masks()
    torch.save(result_masks, "merged.pt")
    print(result_masks)
