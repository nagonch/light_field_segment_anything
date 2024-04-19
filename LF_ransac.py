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
    if not displacement:
        central_mask_centroid = torch.tensor(binary_mask_centroid(mask_central)).cuda()
        epipolar_line_point = torch.tensor(binary_mask_centroid(mask_subview)).cuda()
        displacement = project_point_onto_line(
            epipolar_line_point, epipolar_line_vector, central_mask_centroid
        )
    vec = torch.round(epipolar_line_vector * displacement).long()
    mask_new = shift_binary_mask(mask_subview, vec)
    iou = metric(mask_central, mask_new).item()
    return iou, torch.abs(displacement)


def fit_point_to_LF(segments, segment_num, point, disparity=None):
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
    mask_central = (segments == segment_num)[s_central, t_central]
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
                displacement=disparity,
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


def fit_points_to_LF(segments, segment_num, point, eval=False):
    disparities = []
    ious = []
    matches = []
    for point in point:
        match, disparity, max_iou = fit_point_to_LF(segments, segment_num, point)
        disparities.append(disparity)
        ious.append(max_iou)
        matches.append(match)
    disparities_estimates = torch.stack(disparities).cuda()
    ious = torch.stack([torch.tensor(x) for x in ious]).cuda()
    ious_estimates = F.normalize(ious[None], p=1)[0]
    disparity_parameter = (
        disparities_estimates * ious_estimates
    ).sum()  # weighted sum by ious
    if eval:
        result_matches = []
        mean_ious = ious.mean()
        std_ious = ious.std()
        for match, iou in zip(matches, ious):
            if iou >= mean_ious - 3 * std_ious and iou <= mean_ious + 3 * std_ious:
                result_matches.append(match)
        return (
            disparity_parameter,
            result_matches,
        )
    else:
        return disparity_parameter


def get_inliers_for_disparity(segments, segment_num, points, disparity):
    inliers = []
    for point in points:
        _, disparity, max_iou = fit_point_to_LF(segments, segment_num, point, disparity)
        if max_iou >= CONFIG["metric-threshold"]:
            inliers.append(point)
    return inliers


def LF_ransac(
    segments,
    segment_num,
    n_fitting_points=CONFIG["ransac-n-fitting-points"],
    min_inliers=CONFIG["ransac-n-inliers"],
    n_iterations=CONFIG["ransac-n-iterations"],
):
    best_match = []
    best_disparity = torch.nan
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    indices = get_subview_indices(segments.shape[0], segments.shape[1])
    indices = torch.stack(
        [
            ind
            for ind in indices
            if (ind != torch.tensor([s_central, t_central]).cuda()).any()
        ]
    ).cuda()
    for i in range(n_iterations):
        indices_permuted = indices[torch.randperm(indices.shape[0])]
        fitting_points = indices_permuted[:n_fitting_points]
        disparity_parameter = fit_points_to_LF(segments, segment_num, fitting_points)
        inlier_points = get_inliers_for_disparity(
            segments,
            segment_num,
            indices_permuted,
            disparity_parameter,
        )
        if len(inlier_points) >= min_inliers:
            disparity_parameter, segment_matches = fit_points_to_LF(
                segments,
                segment_num,
                torch.stack(inlier_points),
                eval=True,
            )
            best_match = segment_matches
            best_disparity = disparity_parameter
            break
    return best_match, best_disparity


if __name__ == "__main__":
    segments = torch.tensor(torch.load("segments.pt")).cuda()
    # print(torch.unique(segments[4, 4]))
    # raise
    print(LF_ransac(segments, 331026))
