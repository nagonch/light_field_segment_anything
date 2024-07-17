from utils import visualize_segmentation_mask, EXP_CONFIG
import os
from experiments import get_datset
from plenpy.lightfields import LightField
import torch
import argparse


def visualize(frame_number):
    dataset = get_datset()
    LF_data, _, _ = dataset[frame_number]
    print("visualizing light field image")
    LF = LightField(LF_data)
    LF.show()
    frame_number = str(frame_number).zfill(4)
    exp_name = EXP_CONFIG["exp-name"]

    merged_segments_filename = f"experiments/{exp_name}/{frame_number}_result.pth"
    if os.path.exists(merged_segments_filename):
        merged_segments = torch.load(merged_segments_filename).cpu().numpy()
        print("visualizing merged segments")
        visualize_segmentation_mask(merged_segments, LF_data)
        visualize_segmentation_mask(merged_segments, LF_data, only_boundaries=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_number", type=int)
    args = parser.parse_args()
    visualize(args.frame_number)
