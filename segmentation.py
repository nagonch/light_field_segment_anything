import numpy as np
import torch
import os
from utils import (
    visualize_segments,
    SAM_CONFIG,
    MERGER_CONFIG,
    visualize_segmentation_mask,
)
from utils import stack_segments
from LF_SAM import get_sam
from plenpy.lightfields import LightField
import logging
from data import UrbanLFDataset, HCIOldDataset
from LF_segment_merger import LF_segment_merger

logging.getLogger("plenpy").setLevel(logging.WARNING)


def post_process_segments(segments):
    s, t, u, v = segments.shape
    result_segments = []
    min_mask_area = int(MERGER_CONFIG["min-mask-area-final"] * u * v)
    for i in np.unique(segments)[1:]:
        seg_i = segments == i
        seg_sum = seg_i.sum(axis=(2, 3))
        if (
            seg_sum.mean() >= min_mask_area
            and (seg_sum > min_mask_area).sum() / (s * t)
            >= MERGER_CONFIG["subview-percentage"]
        ):
            result_segments.append(seg_i)
    return result_segments


def main(
    LF,
    segments_filename=SAM_CONFIG["segments-filename"],
    merged_filename=MERGER_CONFIG["merged-filename"],
    segments_checkpoint=SAM_CONFIG["sam-segments-checkpoint"],
    merged_checkpoint=MERGER_CONFIG["merged-checkpoint"],
):
    LF_viz = LightField(LF)
    # LF_viz.show()
    if not (segments_checkpoint and os.path.exists(segments_filename)):
        simple_sam = get_sam()
        simple_sam.segment_LF(LF)
        simple_sam.postprocess_data()
        del simple_sam
        torch.cuda.empty_cache()
    segments = torch.load(segments_filename).cuda()
    # visualize_segmentation_mask(segments.detach().cpu().numpy(), LF)
    if merged_checkpoint and os.path.exists(merged_filename):
        merged_segments = torch.load(merged_filename)
    else:
        merger = LF_segment_merger(segments, torch.load("embeddings.pt"), LF)
        merged_segments = merger.get_result_masks().detach().cpu().numpy()
    # merged_segments = post_process_segments(merged_segments)
    # merged_segments = stack_segments(merged_segments)
    visualize_segmentation_mask(merged_segments, LF)
    torch.save(merged_segments, merged_filename)
    for i, segment in enumerate(np.unique(merged_segments)):
        visualize_segments(
            (merged_segments == segment).astype(np.uint32),
            f"imgs/{str(i).zfill(3)}.png",
        )
    return merged_segments


if __name__ == "__main__":
    dataset = UrbanLFDataset("val")
    LF = dataset[3][0].detach().cpu().numpy()
    # dataset = HCIOldDataset()
    # LF = dataset.get_scene("papillon")
    LF_vis = LightField(LF)
    segments = main(LF)
