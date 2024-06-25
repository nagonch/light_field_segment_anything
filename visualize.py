from utils import visualize_segmentation_mask, EXP_CONFIG
import os
from experiments import get_datset
from plenpy.lightfields import LightField
import torch
import pandas as pd
from metrics import AccuracyMetrics


def visualize(frame_number):
    dataset = get_datset()
    LF, labels, _ = dataset[frame_number]
    print("visualizing light field image")
    LF = LightField(LF)
    LF.show()
    print("visualizing ground truth")
    visualize_segmentation_mask(labels)
    frame_number = str(frame_number).zfill(4)
    exp_name = EXP_CONFIG["exp-name"]
    sam_segments_filename = f"experiments/{exp_name}/{frame_number}_sam_seg.pth"
    if os.path.exists(sam_segments_filename):
        sam_segments = torch.load(sam_segments_filename).cpu().numpy()
        print("visualizing sam segments")
        visualize_segmentation_mask(sam_segments)

    merged_segments_filename = f"experiments/{exp_name}/{frame_number}_result.pth"
    if os.path.exists(merged_segments_filename):
        merged_segments = torch.load(merged_segments_filename).cpu().numpy()
        metrics = AccuracyMetrics(merged_segments, labels)
        _, vis = metrics.achievable_accuracy()
        vis = vis.cpu().numpy()
        print("visualizing merged segments")
        visualize_segmentation_mask(merged_segments)
        uncovered_segments = merged_segments < 1
        print("visualizing uncovered segments")
        visualize_segmentation_mask(uncovered_segments)
        print("visualizing achievable accuracy")
        visualize_segmentation_mask(vis)


def get_metrics_df():
    print("printing metrics:")
    if os.path.exists(f"experiments/{EXP_CONFIG['exp-name']}/metrics.csv"):
        df = pd.read_csv(
            f"experiments/{EXP_CONFIG['exp-name']}/metrics.csv", index_col=0
        )
        print(f"{df}\n\n")


if __name__ == "__main__":
    get_metrics_df()
    visualize(2)
