import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms.functional import affine
from utils import shift_binary_mask


def calculate_peak_metric(segments, i_central, i_subview, step=0.5, eps=1e-9):
    u, v = segments.shape[-2:]
    mask_central = (segments == i_central).to(torch.int32).sum(axis=(0, 1))
    mask_subview = (segments == i_subview).to(torch.int32).sum(axis=(0, 1))
    s_subview, t_subview = torch.where(
        (segments == i_subview).to(torch.int32).sum(axis=(2, 3)) > 0
    )
    s_subview, t_subview = s_subview[0].item(), t_subview[0].item()
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    epipolar_line = (
        torch.tensor([s_central - s_subview, t_central - t_subview]).float().cuda()
    )
    aspect_ratio_matrix = torch.diag(torch.tensor([v, u])).float().cuda()
    epipolar_line = aspect_ratio_matrix @ epipolar_line
    epipolar_line = F.normalize(epipolar_line[None])[0]
    mask_subview_cetroid = (
        torch.stack(torch.where(mask_subview == 1)).float().mean(axis=-1)
    )
    print(mask_subview_cetroid)
    print(epipolar_line)
    # linspace =


# def calculate_shifted_metric(
#     segments,
#     main_segment_i,
#     sub_segment_i,
#     uv_shift,
#     metric=BinaryJaccardIndex().cuda(),
# ):
#     mask_i = (segments == main_segment_i).to(torch.int32).sum(axis=(0, 1))
#     mask_j = (segments == sub_segment_i).to(torch.int32).sum(axis=(0, 1))
#     plt.imshow(mask_i.detach().cpu().numpy(), cmap="gray")
#     plt.savefig("mask_from.png")
#     plt.close()
#     plt.imshow(mask_j.detach().cpu().numpy(), cmap="gray")
#     plt.savefig("mask_to.png")
#     plt.close()
#     new_mask_j = shift_binary_mask(mask_j, uv_shift)
#     result_metric = metric(mask_i, new_mask_j).item()
#     return result_metric


if __name__ == "__main__":
    segments = torch.tensor(torch.load("segments.pt")).cuda()
    segment_central = 31804
    segment_subview = 597
    calculate_peak_metric(segments, segment_central, segment_subview)
