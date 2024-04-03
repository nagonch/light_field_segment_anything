import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms.functional import affine
from utils import shift_binary_mask


def calculate_shifted_metric(
    segments,
    main_segment_i,
    sub_segment_i,
    uv_shift,
    metric=BinaryJaccardIndex().cuda(),
):
    u, v = uv_shift
    mask_i = (segments == main_segment_i).to(torch.int32).sum(axis=(0, 1))
    mask_j = (segments == sub_segment_i).to(torch.int32).sum(axis=(0, 1))
    new_mask_j = shift_binary_mask(mask_j, u, v)
    result_metric = metric(mask_i, new_mask_j).item()
    return result_metric


if __name__ == "__main__":
    segments = torch.load("segments.pt").cuda()
    shifted_metric = calculate_shifted_metric(
        segments,
        177974,
        4682,
        (-20, 0),
    )
