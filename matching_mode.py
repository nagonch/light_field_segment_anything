from sam2_functions import get_auto_mask_predictor, generate_image_masks
from data import HCIOldDataset
import warnings
from utils import visualize_segmentation_mask
import torch

warnings.filterwarnings("ignore")


# move to utils later
def reduce_masks(masks, offset):
    areas = masks.sum(dim=(1, 2))
    masks_result = torch.zeros_like(masks[0]).long()
    for i, _ in enumerate(torch.argsort(areas, descending=True)):
        masks_result[masks[i]] += i
        masks_result = torch.clip(masks_result, 0, i).long()
    masks_result[masks_result != 0] += offset
    return masks_result


def segment_subviews(mask_predictor, LF):
    s_size, t_size, u_size, v_size = LF.shape[:-1]
    result = torch.zeros((s_size, t_size, u_size, v_size)).long().cuda()
    offset = 0
    for s in range(s_size):
        for t in range(t_size):
            subview = LF[s, t]
            masks = generate_image_masks(mask_predictor, subview).bool()
            masks = reduce_masks(
                masks,
                offset,
            )
            offset = masks.max()
            result[s][t] = masks
            del masks
    return result


if __name__ == "__main__":
    mask_predictor = get_auto_mask_predictor()
    dataset = HCIOldDataset()
    for LF, _, _ in dataset:
        LF = LF[3:-3, 3:-3]
        segmentation = segment_subviews(mask_predictor, LF).cpu().numpy()
        visualize_segmentation_mask(segmentation, LF)
