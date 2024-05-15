import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import binary_mask_centroid, get_subview_indices, test_mask, CONFIG


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
    def test_mask(self, mask_num, s, t, p, v):
        mask = self.segments[s, t] == mask_num
        v_len = torch.norm(v)
        u_mask, v_mask = torch.where(mask == 1)
        error = torch.abs(v[0] * (u_mask - p[0]) + v[1] * (v_mask - p[1])) / v_len
        return error.min() <= CONFIG["mask-test-threshold"]

    @torch.no_grad()
    def filter_segments(self, subview_segments, central_mask_centroid, s, t):
        subview_segments = subview_segments[
            ~torch.isin(subview_segments, torch.tensor(self.merged_segments).cuda())
        ]
        subview_segments = torch.stack(
            [
                num
                for num in subview_segments
                if self.test_mask(
                    num, s, t, central_mask_centroid, self.epipolar_line_vectors[s, t]
                )
            ]
        )
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
        result = torch.stack(segments_embeddings).cuda()
        segment_nums_filtered = torch.stack(segment_nums_filtered).cuda()
        return result

    @torch.no_grad()
    def fit(self, central_mask_num, central_mask_centroid, s, t):
        # TODO: embeddings define similarity in texture, iou in shape. Integrate iou
        subview_segments = torch.unique(self.segments[s, t])[1:]
        subview_segments = self.filter_segments(
            subview_segments, central_mask_centroid, s, t
        )
        central_embedding = self.embeddings[central_mask_num.item()][0][None]
        embeddings = self.get_segments_embeddings(subview_segments)
        central_embedding = torch.repeat_interleave(
            central_embedding, embeddings.shape[0], dim=0
        )
        similarities = F.cosine_similarity(embeddings, central_embedding)
        result_segment_index = torch.argmax(similarities).item()
        result_segment = subview_segments[result_segment_index]
        # Get the depth of the segment
        result_similarity = similarities[result_segment_index]
        result_centroid = binary_mask_centroid(self.segments[s, t] == result_segment)
        result_depth = torch.norm(result_centroid - central_mask_centroid)
        return result_segment, result_depth, result_similarity

    @torch.no_grad()
    def predict(self, central_mask_centroid, s, t, depth):
        subview_segments = torch.unique(self.segments[s, t])[1:]
        subview_segments = self.filter_segments(
            subview_segments, central_mask_centroid, s, t
        )
        centroids = torch.stack(
            [
                binary_mask_centroid(self.segments[s, t] == segment)
                for segment in subview_segments
            ]
        ).cuda()
        target_point = central_mask_centroid + self.epipolar_line_vectors[s, t] * depth
        target_point = target_point.repeat(centroids.shape[0], 1)
        distances = torch.norm(centroids - target_point, dim=1)
        result_segment_index = torch.argmin(distances).item()
        result_segment = subview_segments[result_segment_index]
        return result_segment

    @torch.no_grad()
    def find_matches(self, central_mask_num):
        matches = []
        central_mask = (self.segments == central_mask_num)[
            self.s_central, self.t_central
        ]
        central_mask_centroid = binary_mask_centroid(central_mask)
        # 1. Sample a random s, t
        indices_shuffled = self.shuffle_indices()
        s_main, t_main = indices_shuffled[0]
        # 2. Find a segment match and a depth "the hard way"
        matched_segment, depth, certainty = self.fit(
            central_mask_num, central_mask_centroid, s_main, t_main
        )
        for s, t in indices_shuffled:
            match = self.predict(central_mask_centroid, s, t, depth)
            matches.append(match)
        # 3. For the rest of s and t find match a closest to the depth using centroids
        return matches

    @torch.no_grad()
    def get_result_masks(self):
        self.merged_segments = []
        for segment_num in tqdm(self.central_segments):
            segment_embedding = self.embeddings.get(segment_num.item(), None)
            if segment_embedding is None:
                continue
            matches = self.find_matches(segment_num)
            self.segments[torch.isin(self.segments, torch.tensor(matches).cuda())] = (
                segment_num
            )
            self.merged_segments.append(segment_num)
        return self.segments


if __name__ == "__main__":
    segments = torch.load("segments.pt").cuda()
    embeddings = torch.load("embeddings.pt")
    merger = LF_RANSAC_segment_merger(segments, embeddings)
    result_masks = merger.get_result_masks()
    print(result_masks)
