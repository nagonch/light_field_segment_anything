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
    mask_i = (segments == main_segment_i).to(torch.int32).sum(axis=(0, 1))
    mask_j = (segments == sub_segment_i).to(torch.int32).sum(axis=(0, 1))
    plt.imshow(mask_i.detach().cpu().numpy(), cmap="gray")
    plt.savefig("mask_from.png")
    plt.close()
    plt.imshow(mask_j.detach().cpu().numpy(), cmap="gray")
    plt.savefig("mask_to.png")
    plt.close()
    new_mask_j = shift_binary_mask(mask_j, uv_shift)
    result_metric = metric(mask_i, new_mask_j).item()
    return result_metric


if __name__ == "__main__":
    segments = torch.tensor(torch.load("merged.pt")).cuda()
    segments[segments != 178183] = 0
    num = 1
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            segments[i, j][segments[i, j] != 0] += num
            num += 1
    # print(torch.unique(segments))
    # raise
    shifted_metric = calculate_shifted_metric(
        segments,
        178184,
        178191,
        (0, 0),
    )
    print(shifted_metric)
