import torch
import torch.nn.functional as F
from utils import get_subview_indices
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex
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


def calculate_peak_metric(
    mask_central,
    mask_subview,
    epipolar_line_vector,
    displacement=None,
    metric=BinaryJaccardIndex().cuda(),
):
    central_mask_centroid = torch.tensor(binary_mask_centroid(mask_central)).cuda()
    epipolar_line_point = torch.tensor(binary_mask_centroid(mask_subview)).cuda()
    displacement = project_point_onto_line(
        epipolar_line_point, epipolar_line_vector, central_mask_centroid
    )
    vec = torch.round(epipolar_line_vector * displacement).long()
    mask_new = shift_binary_mask(mask_subview, vec)
    iou = metric(mask_central, mask_new).item()
    return iou, torch.abs(displacement)


def fit_point_to_LF(segments, central_segment_num, point):
    s, t = point
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
    mask_central = (segments == central_segment_num)[s_central, t_central]
    epipolar_line_point = binary_mask_centroid(mask_central)
    line_boundries = line_image_boundaries(
        epipolar_line_point, epipolar_line_vector, u, v
    )
    epipolar_line = draw_line_in_mask(
        torch.zeros_like(mask_central),
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
                mask_central,
                seg,
                epipolar_line_vector,
            )
            max_ious_result.append(max_iou)
            disparities.append(disparity)
    if not segments_result:
        return (-1, 0, 0)  # match not found
    best_match_indx = np.argmax(max_ious_result)
    return (
        segments_result[best_match_indx],
        disparities[best_match_indx],
        max_ious_result[best_match_indx],
    )


def LF_ransac(
    segments,
    segment_num,
    n_fitting_points=2,
    n_iterations=20,
    error_threshold=0.05,
    min_inliers=50,
):
    best_error = torch.inf
    best_match = []
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    indices = get_subview_indices(segments.shape[0], segments.shape[1])
    indices = torch.stack(
        [ind for ind in indices if (ind != torch.tensor([s_central, t_central])).any()]
    )
    for i in range(n_iterations):
        indices_permuted = indices[torch.randperm(indices.shape[0])]
        fitting_points = indices_permuted[:n_fitting_points]
        testing_points = indices_permuted[n_fitting_points:]
        for point in fitting_points:
            segment, disparity, max_iou = fit_point_to_LF(segments, segment_num, point)
            print(segment, disparity, max_iou)


if __name__ == "__main__":
    segments = torch.tensor(torch.load("segments.pt")).cuda()
    LF_ransac(segments, 331232)
