import torch
import torch.nn.functional as F
from tqdm import tqdm


class LF_RANSAC_segment_merger:
    @torch.no_grad()
    def __init__(self, segments, embeddings):
        self.segments = segments
        self.embeddings = embeddings
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2
        self.central_cegments = self.get_central_segments()

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
    def find_matches(self, central_mask_num):
        matches = []
        return matches

    @torch.no_grad()
    def get_result_masks(self):
        self.merged_segments = []
        for segment_num in tqdm(self.central_cegments):
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
