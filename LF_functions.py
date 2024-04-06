import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import shift_binary_mask


def calculate_peak_metric(
    segments,
    i_central,
    i_subview,
    n_points=100,
    metric=BinaryJaccardIndex().cuda(),
):
    u, v = segments.shape[-2:]
    mask_central = (segments == i_central).to(torch.int32).sum(axis=(0, 1))
    mask_subview = (segments == i_subview).to(torch.int32).sum(axis=(0, 1))
    # plt.imshow(mask_central.detach().cpu().numpy())
    # plt.show()
    # plt.savefig("central.png")
    # plt.close()
    # plt.imshow(mask_subview.detach().cpu().numpy())
    # plt.show()
    # plt.savefig("subview.png")
    # plt.close()
    s_subview, t_subview = torch.where(
        (segments == i_subview).to(torch.int32).sum(axis=(2, 3)) > 0
    )
    s_subview, t_subview = s_subview[0].item(), t_subview[0].item()
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    epipolar_line_vector = (
        torch.tensor([s_central - s_subview, t_central - t_subview]).float().cuda()
    )
    aspect_ratio_matrix = torch.diag(torch.tensor([v, u])).float().cuda()
    epipolar_line_vector = aspect_ratio_matrix @ epipolar_line_vector
    epipolar_line_vector = F.normalize(epipolar_line_vector[None])[0]
    magnitudes = torch.linspace(-500, 500, n_points).cuda()
    ious = []
    for i, magnitude in enumerate(magnitudes):
        vec = torch.round(epipolar_line_vector * magnitude).long()
        mask_new = shift_binary_mask(mask_subview, vec)
        # plt.imshow(mask_new.detach().cpu().numpy())
        # plt.show()
        # plt.savefig(f"masks/{str(i).zfill(3)}.png")
        # plt.close()
        iou = metric(mask_central, mask_new)
        ious.append(iou.item())
    return iou
    # plt.plot(ious)
    # plt.show()
    # plt.savefig("ious.png")
    # print(torch.tensor(ious).max())
    # epipolar_line_point = (
    #     torch.stack(torch.where(mask_subview == 1)).float().mean(axis=-1)
    # )
    # u_space = torch.linspace(
    #     0, epipolar_line_point[0] / epipolar_line_vector[0], n_points
    # ).cuda()
    # v_space = (
    #     (u_space - epipolar_line_point[0])
    #     * epipolar_line_vector[1]
    #     / epipolar_line_vector[0]
    # ) + epipolar_line_point[1]
    # print(epipolar_line_point)
    # plt.scatter(u_space.detach().cpu().numpy(), v_space.detach().cpu().numpy())
    # plt.show()
    # plt.savefig("yo.png")
    # linspace =


if __name__ == "__main__":
    from random import randint

    segments = torch.tensor(torch.load("segments.pt")).cuda()
    unique_segments = torch.unique(segments[2, 2])
    segment_central = unique_segments[randint(0, unique_segments.shape[0]) - 1]
    print(segment_central)
    segment_subview = 15397
    calculate_peak_metric(segments, segment_central, segment_subview)
