import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import shift_binary_mask


def project_point_onto_line(x, v, y):
    # Calculate the vector from x to y
    a = y - x

    # Calculate the projection of a onto v
    projection = torch.dot(a, v) / torch.dot(v, v) * v

    # Calculate the scalar t
    t = torch.dot(projection, v) / torch.dot(v, v)

    return t


def calculate_peak_metric(
    segments,
    i_central,
    i_subview,
    metric=BinaryJaccardIndex().cuda(),
):
    u, v = segments.shape[-2:]
    mask_central = (segments == i_central).to(torch.int32).sum(axis=(0, 1))
    mask_subview = (segments == i_subview).to(torch.int32).sum(axis=(0, 1))

    s_subview, t_subview = torch.where(
        (segments == i_subview).to(torch.int32).sum(axis=(2, 3)) > 0
    )  # define the index of the subview
    s_subview, t_subview = (
        s_subview[0].item(),
        t_subview[0].item(),
    )  # define the index of the central subview
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    epipolar_line_vector = (
        torch.tensor([s_central - s_subview, t_central - t_subview]).float().cuda()
    )  # the direction of the epipolar line in this subview
    aspect_ratio_matrix = (
        torch.diag(torch.tensor([v, u])).float().cuda()
    )  # in case the image is non-square
    epipolar_line_vector = aspect_ratio_matrix @ epipolar_line_vector
    epipolar_line_vector = F.normalize(epipolar_line_vector[None])[0]
    central_mask_x, central_mask_y = torch.where(mask_central == 1)
    central_mask_centroid = torch.tensor(
        [
            central_mask_x.float().mean(),
            central_mask_y.float().mean(),
        ]
    ).cuda()
    subview_mask_x, subview_mask_y = torch.where(mask_subview == 1)
    epipolar_line_point = torch.tensor(
        [
            subview_mask_x.float().mean(),
            subview_mask_y.float().mean(),
        ]
    ).cuda()
    displacement = project_point_onto_line(
        epipolar_line_point, epipolar_line_vector, central_mask_centroid
    )
    vec = torch.round(epipolar_line_vector * displacement).long()
    mask_new = shift_binary_mask(mask_subview, vec)
    iou = metric(mask_central, mask_new).item()
    return iou


if __name__ == "__main__":
    from random import randint

    segments = torch.tensor(torch.load("segments.pt")).cuda()
    unique_central_segments = torch.unique(segments[2, 2])
    unique_corner_segments = torch.unique(segments[0, 0])
    # segment_central = 32723
    segment_central = unique_central_segments[
        randint(0, unique_central_segments.shape[0]) - 1
    ]
    print(segment_central)
    segment_subview = unique_corner_segments[
        randint(0, unique_corner_segments.shape[0]) - 1
    ]
    print(segment_subview)
    print(calculate_peak_metric(segments, segment_central, segment_subview))
