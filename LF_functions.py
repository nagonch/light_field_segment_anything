import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import shift_binary_mask


def calculate_peak_metric(
    segments,
    i_central,
    i_subview,
    pixel_step=50,
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
    ious = [0]
    mask_new = torch.ones_like(mask_subview)
    i = 0
    # shift the segment along the line until it disappears, calculate iou
    while mask_new.sum() > 0:
        vec = torch.round(epipolar_line_vector * i * -pixel_step).long()
        mask_new = shift_binary_mask(mask_subview, vec)
        iou = metric(mask_central, mask_new)
        if iou < ious[-1]:
            break
        ious.append(iou.item())
        i += 1
    mask_new = torch.ones_like(mask_subview)
    i = 0
    while mask_new.sum() > 0:
        vec = torch.round(epipolar_line_vector * i * pixel_step).long()
        mask_new = shift_binary_mask(mask_subview, vec)
        iou = metric(mask_central, mask_new)
        if iou < ious[-1]:
            break
        ious.append(iou.item())
        i += 1
    result = torch.max(torch.tensor(ious)).item()
    return result


if __name__ == "__main__":
    from random import randint

    segments = torch.tensor(torch.load("segments.pt")).cuda()
    unique_segments = torch.unique(segments[2, 2])
    segment_central = 32723
    # segment_central = unique_segments[randint(0, unique_segments.shape[0]) - 1]
    print(segment_central)
    segment_subview = 15397
    print(calculate_peak_metric(segments, segment_central, segment_subview))
