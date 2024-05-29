import torch
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
        self.verbose = MERGER_CONFIG["verbose"]
        self.embedding_coeff = MERGER_CONFIG["embedding-coeff"]
        if self.verbose:
            os.makedirs("LF_ransac_output", exist_ok=True)

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
        self.segments[
            ~torch.isin(
                self.segments,
                torch.unique(self.segments[self.s_central, self.t_central]),
            )
        ] = 0  # TODO: check what happens with unmatched
        return self.segments


if __name__ == "__main__":
    pass
