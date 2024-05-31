import numpy as np
import torch
import os
from utils import (
    visualize_segments,
    SAM_CONFIG,
    MERGER_CONFIG,
    save_LF_image,
    visualize_segmentation_mask,
)
from data import get_LF
from sam_functions import get_sam
from plenpy.lightfields import LightField
import logging
from scipy.io import loadmat
from data import LFDataset
from LF_segment_merger import LF_segment_merger

logging.getLogger("plenpy").setLevel(logging.WARNING)


def post_process_segments(segments):
    u, v = segments.shape[-2:]
    result_segments = []
    min_mask_area = int(MERGER_CONFIG["min-mask-area-final"] * u * v)
    for i in np.unique(segments)[1:]:
        seg_i = segments == i
        if seg_i.sum(axis=(2, 3)).mean() >= min_mask_area:
            result_segments.append(seg_i)
    return result_segments


def main(
    LF,
    segments_filename=SAM_CONFIG["segments-filename"],
    merged_filename=MERGER_CONFIG["merged-filename"],
    segments_checkpoint=SAM_CONFIG["sam-segments-checkpoint"],
    merged_checkpoint=MERGER_CONFIG["merged-checkpoint"],
):
    if segments_checkpoint and os.path.exists(segments_filename):
        segments = torch.load(segments_filename).cuda()
    else:
        simple_sam = get_sam()
        segments = simple_sam.segment_LF(LF)
        simple_sam.postprocess_embeddings()
        torch.save(segments, segments_filename)
        del simple_sam
        torch.cuda.empty_cache()
    if merged_checkpoint and os.path.exists(merged_filename):
        merged_segments = torch.load(merged_filename).detach().cpu().numpy()
    else:
        merger = LF_segment_merger(
            torch.clone(segments), torch.load("embeddings.pt"), LF
        )
        merged_segments = merger.get_result_masks().detach().cpu().numpy()
        torch.save(merged_segments, merged_filename)
    LF = LightField(LF)
    LF.show()
    visualize_segmentation_mask(
        segments.detach().cpu().numpy(),
    )
    visualize_segmentation_mask(
        merged_segments,
    )
    merged_segments = post_process_segments(merged_segments)
    for i, segment in enumerate(merged_segments):
        visualize_segments(
            segment.astype(np.uint32),
            f"imgs/{str(i).zfill(3)}.png",
        )
    return merged_segments


if __name__ == "__main__":
    dataset = LFDataset("UrbanLF_Syn/val")
    LF = dataset[3].detach().cpu().numpy()
    LF_vis = LightField(LF)
    segments = main(LF)
