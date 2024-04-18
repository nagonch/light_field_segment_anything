import numpy as np
import torch
import os
from utils import visualize_segments, CONFIG, save_LF_image, visualize_segmentation_mask
from data import get_LF
from sam_functions import get_sam
from LF_functions import LF_segment_merger
from plenpy.lightfields import LightField
import logging

logging.getLogger("plenpy").setLevel(logging.WARNING)


def post_process_segments(segments):
    u, v = segments.shape[-2:]
    result_segments = []
    min_mask_area = int(CONFIG["min-mask-area-final"] * u * v)
    for i in np.unique(segments)[1:]:
        seg_i = segments == i
        if seg_i.sum(axis=(2, 3)).mean() >= min_mask_area:
            result_segments.append(seg_i)
    return result_segments


def main(
    LF_dir,
    segments_filename=CONFIG["segments-filename"],
    embeddings_filename=CONFIG["embeddings-filename"],
    merged_filename=CONFIG["merged-filename"],
    segments_checkpoint=CONFIG["sam-segments-checkpoint"],
    merged_checkpoint=CONFIG["merged-checkpoint"],
    vis_filename=CONFIG["vis-filename"],
):
    from scipy.io import loadmat
    from data import LFDataset

    dataset = LFDataset("UrbanLF_Syn/test")
    LF = dataset[3].detach().cpu().numpy()
    # LF = loadmat("Lego.256.mat")["LF"][4:-4, 4:-4]
    LF_vis = LightField(LF)
    LF_vis.show()
    save_LF_image(np.array(LF), "input_LF.png")
    # LF = get_LF(LF_dir)
    # LF = loadmat("lego_128.mat")["LF"].astype(np.int32)[1:-1, 1:-1]
    simple_sam = get_sam()
    if segments_checkpoint and os.path.exists(segments_filename):
        segments = torch.load(segments_filename).cuda()
        embeddings = torch.load(embeddings_filename).cuda()
    else:
        segments = simple_sam.segment_LF(LF)
        simple_sam.postprocess_embeddings()
        torch.save(segments, segments_filename)
    if merged_checkpoint and os.path.exists(merged_filename):
        segments = torch.load(merged_filename)
    else:
        segment_merger = LF_segment_merger(segments)
        segments = segment_merger.get_result_masks().detach().cpu().numpy()
        torch.save(segments, merged_filename)
    visualize_segmentation_mask(
        segments,
        vis_filename,
    )
    segments = post_process_segments(segments)
    for i, segment in enumerate(segments):
        visualize_segments(
            segment.astype(np.uint32),
            f"imgs/{str(i).zfill(3)}.png",
        )
    return segments


if __name__ == "__main__":
    dir = "/home/cedaradmin/blender/lightfield/LFPlane/f00051/png"
    segments = main(dir)
