import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import binary_mask_centroid, get_subview_indices


class LF_RANSAC_segment_merger:
    @torch.no_grad()
    def __init__(self, segments, embeddings):
        self.segments = segments
        self.embeddings = embeddings
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2
        self.subview_indices = get_subview_indices(self.s_size, self.t_size)
        self.central_segments = self.get_central_segments()

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
        # 3. For the rest of s and t find match a closest to the depth using centroids
        return matches

    @torch.no_grad()
    def get_result_masks(self):
        self.merged_segments = []
        for segment_num in tqdm(self.central_segments):
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
