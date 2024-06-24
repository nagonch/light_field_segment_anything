from utils import visualize_segmentation_mask, EXP_CONFIG
import os
from experiments import get_datset
from plenpy.lightfields import LightField
import torch


def visualize(frame_number):
    dataset = get_datset()
    LF, labels, _ = dataset[frame_number]
    LF = LightField(LF)
    LF.show()
    labels = labels
    visualize_segmentation_mask(labels)
    frame_number = str(frame_number).zfill(4)
    exp_name = EXP_CONFIG["exp-name"]
    sam_segments_filename = f"experiments/{exp_name}/{frame_number}_sam_seg.pth"
    if os.path.exists(sam_segments_filename):
        sam_segments = torch.load(sam_segments_filename).cpu().numpy()
        visualize_segmentation_mask(sam_segments)
    merged_segments_filename = f"experiments/{exp_name}/{frame_number}_result.pth"
    if os.path.exists(merged_segments_filename):
        merged_segments = torch.load(merged_segments_filename).cpu().numpy()
        visualize_segmentation_mask(merged_segments)


if __name__ == "__main__":
    visualize(0)
