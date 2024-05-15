import torch
import torch.nn.functional as F


class LF_RANSAC_segment_merger:
    @torch.no_grad()
    def __init__(self, segments, embeddings):
        self.segments = segments
        self.embeddings = embeddings
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2

    @torch.no_grad()
    def get_result_masks(self):
        return self.segments


if __name__ == "__main__":
    pass
