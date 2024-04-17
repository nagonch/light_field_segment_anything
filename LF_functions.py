import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from utils import (
    shift_binary_mask,
    project_point_onto_line,
    draw_line_in_mask,
    line_image_boundaries,
    binary_mask_centroid,
    visualize_segments,
    get_subview_indices,
    CONFIG,
)
from tqdm import tqdm


def calculate_peak_metric(
    mask_central,
    central_mask_centroid,
    mask_subview,
    epipolar_line_vector,
    metric=BinaryJaccardIndex().cuda(),
):
    epipolar_line_point = torch.tensor(binary_mask_centroid(mask_subview)).cuda()
    displacement = project_point_onto_line(
        epipolar_line_point, epipolar_line_vector, central_mask_centroid
    )
    vec = torch.round(epipolar_line_vector * displacement).long()
    mask_new = shift_binary_mask(mask_subview, vec)
    iou = metric(mask_central, mask_new).item()
    return iou, torch.abs(displacement)


def find_match(main_mask, main_mask_centroid, s, t):
    u, v = segments.shape[-2:]
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    epipolar_line_vector = (
        torch.tensor([s_central - s, t_central - t]).float().cuda()
    )  # the direction of the epipolar line in this subview
    aspect_ratio_matrix = (
        torch.diag(torch.tensor([v, u])).float().cuda()
    )  # in case the image is non-square
    epipolar_line_vector = aspect_ratio_matrix @ epipolar_line_vector
    epipolar_line_vector = F.normalize(epipolar_line_vector[None])[0]
    line_boundries = line_image_boundaries(
        main_mask_centroid.detach().cpu().numpy(), epipolar_line_vector, u, v
    )
    epipolar_line = draw_line_in_mask(
        torch.zeros_like(main_mask),
        line_boundries[0],
        line_boundries[1],
    )
    segments_result = []
    max_ious_result = []
    disparities = []
    for segment_num in torch.unique(segments[s, t])[1:]:
        seg = segments[s, t] == segment_num
        if torch.max(seg.to(torch.int32) + epipolar_line.to(torch.int32)) > 1:
            segments_result.append(segment_num.item())
            max_iou, disparity = calculate_peak_metric(
                main_mask,
                main_mask_centroid,
                seg,
                epipolar_line_vector,
            )
            max_ious_result.append(max_iou)
            disparities.append(disparity)
    max_iou = np.max(max_ious_result)
    if not segments_result or max_iou <= CONFIG["metric-threshold"]:
        return -1  # match not found
    return segments_result[np.argmax(max_ious_result)]


def get_epipolar_line_vectors(segments):
    s, t, u, v = segments.shape
    s_central, t_central = s // 2, t // 2
    subview_indices = get_subview_indices(s, t)
    epipolar_line_vectors = (
        torch.tensor([s_central, t_central]).cuda() - subview_indices
    ).float()
    aspect_ratio_matrix = (
        torch.diag(torch.tensor([v, u])).float().cuda()
    )  # in case the image is non-square
    epipolar_line_vectors = (aspect_ratio_matrix @ epipolar_line_vectors.T).T
    epipolar_line_vectors = F.normalize(epipolar_line_vectors)
    epipolar_line_vectors = epipolar_line_vectors.reshape(s, t, 2)
    return epipolar_line_vectors


def find_matches(segments, segment_num):
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    main_mask = (segments == segment_num)[s_central, t_central]
    main_mask_centroid = torch.tensor(binary_mask_centroid(main_mask)).cuda()
    matches = []
    for s in range(segments.shape[0]):
        for t in range(segments.shape[1]):
            if s == s_central and t == t_central:
                continue
            segment_match = find_match(main_mask, main_mask_centroid, s, t)
            if segment_match >= 0:
                matches.append(segment_match)
    return matches


def get_result_masks(segments):
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    central_segments = torch.unique(segments[s_central, t_central])[1:]
    segment_sums = [(segments == i).sum() for i in central_segments]
    central_segments = [
        segment
        for _, segment in sorted(zip(segment_sums, central_segments), reverse=True)
    ]
    for segment_num in tqdm(central_segments):
        matches = find_matches(segments, segment_num)
        segments[torch.isin(segments, torch.tensor(matches).cuda())] = segment_num
    segments[~torch.isin(segments, torch.unique(segments[s_central, t_central]))] = 0
    return segments


class LF_segment_merger:
    @torch.no_grad()
    def __init__(self, segments):
        self.segments = segments
        self.s_size, self.t_size, self.v_size, self.u_size = segments.shape
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
        central_segments = central_segments[torch.argsort(segment_sums)]
        print(central_segments)
        return central_segments

    @torch.no_grad()
    def get_result_masks(self, segments):
        for segment_num in tqdm(self.central_segments):
            matches = find_matches(segments, segment_num)
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
    segment_merger = LF_segment_merger(segments)
    print(segment_merger)
    # find_matches_RANSAC(segments, 331232, n_data_points=70)
    # print(get_result_masks(segments))
    # central_test_segment = 32862
    # print(get_segmentation(segments, central_test_segment))
