from utils import visualize_segmentation_mask, EXP_CONFIG
import os
from experiments import get_datset
from plenpy.lightfields import LightField
import torch
import argparse
from utils import vis_to_gif


def visualize(frame_number):
    dataset = get_datset()
    LF_data, _, _ = dataset[frame_number]
    print("visualizing light field image")
    LF = LightField(LF_data)
    # LF.show()
    frame_number = str(frame_number).zfill(4)
    exp_name = EXP_CONFIG["exp-name"]

    merged_segments_filename = f"experiments/{exp_name}/{frame_number}_result.pth"
    if os.path.exists(merged_segments_filename):
        merged_segments = torch.load(merged_segments_filename).cpu().numpy()
        print("visualizing merged segments")
        vis = visualize_segmentation_mask(merged_segments, LF_data, just_return=True)
        # visualize_segmentation_mask(merged_segments, LF_data, only_boundaries=True)
        return vis


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("frame_number", type=int)
    # args = parser.parse_args()
    gifs_dir = f'{EXP_CONFIG["exp-name"]}_gifs'
    os.makedirs(gifs_dir, exist_ok=True)
    for i in range(28):
        vis = visualize(i)
        vis_to_gif(visualize(i), f"{gifs_dir}/{str(i).zfill(4)}.gif")
        raise
