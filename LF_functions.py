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
)


def calculate_peak_metric(
    mask_central,
    mask_subview,
    epipolar_line_vector,
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
    return iou


def find_match(segments, central_segment_num, s, t):
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
    for segment_num in torch.unique(segments[s, t]):
        seg = segments[s, t] == segment_num
        if torch.max(seg.to(torch.int32) + epipolar_line.to(torch.int32)) > 1:
            segments_result.append(segment_num.item())
            max_ious_result.append(
                calculate_peak_metric(mask_central, seg, epipolar_line_vector)
            )
    return segments_result[np.argmax(max_ious_result)]


if __name__ == "__main__":
    # mask = torch.zeros((256, 341))
    # mask = draw_line_in_mask(mask, (0, 0), (230, 240))
    # plt.imshow(mask)
    # plt.show()
    # plt.close()
    # raise
    from random import randint

    segments = torch.tensor(torch.load("segments.pt")).cuda()
    central_test_segment = 32862
    segment_match = find_match(segments, central_test_segment, 0, 0)
    seg = (segments[0, 0] == segment_match).to(torch.int32) + (
        segments[2, 2] == central_test_segment
    ).to(torch.int32)
    seg2 = (segments[2, 2] == central_test_segment).to(torch.int32)
    plt.imshow(seg2.detach().cpu().numpy(), cmap="gray")
    plt.show()
    plt.close()
