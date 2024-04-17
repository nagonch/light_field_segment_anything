import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from utils import (
    shift_binary_mask,
    project_point_onto_line,
    test_mask,
    binary_mask_centroid,
    visualize_segments,
    get_subview_indices,
    test_mask,
    CONFIG,
)
from tqdm import tqdm


class LF_segment_merger:
    @torch.no_grad()
    def __init__(self, segments):
        self.segments = segments
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2
        self.subview_indices = get_subview_indices(self.s_size, self.t_size)
        self.epipolar_line_vectors = self.get_epipolar_line_vectors()
        self.central_segments = self.get_central_segments()

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
    def calculate_peak_metric(
        self,
        mask_central,
        central_mask_centroid,
        mask_subview,
        epipolar_line_vector,
        metric=BinaryJaccardIndex().cuda(),
    ):
        epipolar_line_point = binary_mask_centroid(mask_subview)
        displacement = project_point_onto_line(
            epipolar_line_point, epipolar_line_vector, central_mask_centroid
        )
        vec = torch.round(epipolar_line_vector * displacement).long()
        mask_new = shift_binary_mask(mask_subview, vec)
        iou = metric(mask_central, mask_new).item()
        return iou

    @torch.no_grad()
    def find_match(self, main_mask, main_mask_centroid, s, t):
        segments_result = []
        max_ious_result = []
        for segment_num in torch.unique(segments[s, t])[1:]:
            seg = segments[s, t] == segment_num
            if test_mask(seg, main_mask_centroid, self.epipolar_line_vectors[s, t]):
                segments_result.append(segment_num.item())
                max_iou = self.calculate_peak_metric(
                    main_mask,
                    main_mask_centroid,
                    seg,
                    self.epipolar_line_vectors[s, t],
                )
                max_ious_result.append(max_iou)
        if not segments_result or np.max(max_ious_result) <= CONFIG["metric-threshold"]:
            return -1  # match not found
        print(len(torch.unique(segments[s, t])[1:]), len(segments_result))
        return segments_result[np.argmax(max_ious_result)]

    @torch.no_grad()
    def find_matches(self, main_mask, main_mask_centroid):
        matches = []
        for s in range(self.s_size):
            for t in range(self.t_size):
                if s == self.s_central and t == self.t_central:
                    continue
                segment_match = self.find_match(main_mask, main_mask_centroid, s, t)
                if segment_match >= 0:
                    matches.append(segment_match)
        return matches

    @torch.no_grad()
    def get_result_masks(self):
        for segment_num in tqdm(self.central_segments):
            main_mask = (self.segments == segment_num)[self.s_central, self.t_central]
            main_mask_centroid = binary_mask_centroid(main_mask)
            matches = self.find_matches(main_mask, main_mask_centroid)
            segments[torch.isin(segments, torch.tensor(matches).cuda())] = segment_num
        segments[
            ~torch.isin(
                segments, torch.unique(segments[self.s_central, self.t_central])
            )
        ] = 0
        return segments


if __name__ == "__main__":
    # mask = torch.zeros((256, 341))
    # mask = draw_line_in_mask(mask, (0, 0), (230, 240))
    # plt.imshow(mask)
    # plt.show()
    # plt.close()
    # raise
    from random import randint

    segments = torch.tensor(torch.load("segments.pt")).cuda()
    merger = LF_segment_merger(segments)
    merger.get_result_masks()
    # segment_merger = LF_segment_merger(segments)
    # print(segment_merger)
    # find_matches_RANSAC(segments, 331232, n_data_points=70)
    # print(get_result_masks(segments))
    # central_test_segment = 32862
    # print(get_segmentation(segments, central_test_segment))
