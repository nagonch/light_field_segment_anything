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


def find_match(segments, central_segment_num, s, t, displacement=None):
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
                displacement=displacement,
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


def find_matches(segments, segment_num):
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    matches = []
    for s in range(segments.shape[0]):
        for t in range(segments.shape[1]):
            if s == s_central and t == t_central:
                continue
            segment_match, _, _ = find_match(segments, segment_num, s, t)
            if segment_match >= 0:
                matches.append(segment_match)
    return matches


def find_matches_RANSAC(
    segments,
    segment_num,
    n_iterations=20,
    n_data_points=2,
    required_inliers=50,
):
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    indices = get_subview_indices(segments.shape[0], segments.shape[1])
    indices = torch.stack(
        [ind for ind in indices if (ind != torch.tensor([s_central, t_central])).any()]
    )
    if n_data_points >= indices.shape[0]:
        raise ValueError(
            f"n_data_points {n_data_points} too many for {indices.shape[0]} subviews"
        )
    best_matches = []
    best_matches_score = 0
    best_disparity = torch.nan
    for i in range(n_iterations):
        ious = []
        disparities = []
        matched_segment_nums = []
        indices_permuted = indices[torch.randperm(indices.shape[0])]
        estimation_points = indices_permuted[:n_data_points]
        fitting_points = indices_permuted[n_data_points:]
        for point in estimation_points:
            matched_segment_num, disparity, iou = find_match(
                segments, segment_num, point[0].item(), point[1].item()
            )
            ious.append(iou)
            matched_segment_nums.append(matched_segment_num)
            disparities.append(disparity)
        disparities_estimates = torch.stack(disparities).cuda()
        ious_estimates = F.normalize(
            torch.stack([torch.tensor(x) for x in ious]).cuda()[None], p=1
        )[0]
        disparity_parameter = (
            disparities_estimates * ious_estimates
        ).sum()  # weighted sum by ious
        for point in fitting_points:
            matched_segment_num, disparity, iou = find_match(
                segments,
                segment_num,
                point[0].item(),
                point[1].item(),
                displacement=disparity_parameter,
            )
            matched_segment_nums.append(matched_segment_num)
            ious.append(iou)
        ious = torch.stack([torch.tensor(x) for x in ious])
        print(ious.mean())
        raise
        # threshold_low =
        # threshold_high =
        ious_inliers = []
        disparities_inliers = []
        matched_segment_nums_inliers = []
        score = -torch.std(torch.stack([torch.tensor(x) for x in ious_inliers]))
        if score > best_matches_score:
            best_matches_score = score
            best_matches = matched_segment_nums
            best_disparity = disparity_parameter
    return best_matches, best_disparity


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


if __name__ == "__main__":
    # mask = torch.zeros((256, 341))
    # mask = draw_line_in_mask(mask, (0, 0), (230, 240))
    # plt.imshow(mask)
    # plt.show()
    # plt.close()
    # raise
    from random import randint

    segments = torch.tensor(torch.load("segments.pt")).cuda()
    find_matches_RANSAC(segments, 331232, n_data_points=70)
    # print(get_result_masks(segments))
    # central_test_segment = 32862
    # print(get_segmentation(segments, central_test_segment))
