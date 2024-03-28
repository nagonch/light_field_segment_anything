import numpy as np

import torch
from tqdm import tqdm
import yaml
import os
import imgviz
from utils import visualize_segments, get_LF, CONFIG
from sam_functions import get_sam
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex
import networkx as nx
import torch.nn.functional as F


def calculate_segments_metric(segments, embeddings, i, j):
    mask_i = (segments == i).to(torch.int32)
    mask_j = (segments == j).to(torch.int32)
    if CONFIG["edge-metric"] == "cosine":
        embs_i = (embeddings * (segments == i)[:, :, :, :, None].to(torch.int32)).mean(
            axis=(0, 1, 2, 3)
        )
        embs_j = (embeddings * (segments == j)[:, :, :, :, None].to(torch.int32)).mean(
            axis=(0, 1, 2, 3)
        )
        metric_val = F.cosine_similarity(embs_i[None], embs_j[None])
    elif CONFIG["edge-metric"] == "iou":
        iou_metric = BinaryJaccardIndex().cuda()
        segments_i = mask_i.sum(axis=(0, 1))
        segments_j = mask_j.sum(axis=(0, 1))
        metric_val = iou_metric(segments_i, segments_j)
    return metric_val.item()


def get_merge_segments_mapping(segments, embeddings):
    metric = BinaryJaccardIndex().cuda()
    s, t = segments.shape[0], segments.shape[1]
    central_inds = torch.unique(segments[s // 2][t // 2].reshape(-1))[1:]
    graphs = []
    mapping = {}
    for s_ in tqdm(range(s)):
        for t_ in range(t):
            if s_ == s // 2 and t_ == t // 2:
                continue
            graph = nx.Graph()
            segments_st = torch.unique(segments[s_][t_].reshape(-1))[1:]
            top_nodes = segments_st.cpu().detach().numpy()
            bottom_nodes = central_inds.cpu().detach().numpy()
            graph.add_nodes_from(
                top_nodes,
                bipartite=0,
            )
            graph.add_nodes_from(
                bottom_nodes,
                bipartite=1,
            )
            for i in central_inds:
                for j in segments_st:
                    metric_val = calculate_segments_metric(segments, embeddings, i, j)
                    graph.add_edges_from(
                        [(j.item(), i.item())], weight=(1 - metric_val)
                    )
            graphs.append(graph)
            max_matching = nx.bipartite.matching.minimum_weight_full_matching(
                graph, top_nodes=top_nodes
            )
            for node in bottom_nodes:
                if node in max_matching.keys():
                    max_matching.pop(node)
            mapping.update(max_matching)
    return mapping


def post_process_segments(segments):
    u, v = segments.shape[-2:]
    result_segments = []
    min_mask_area = int(CONFIG["min-mask-area"] * u * v)
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
    vis_filename=CONFIG["vis-filename"],
):
    LF = get_LF(LF_dir)
    simple_sam = get_sam()
    if segments_checkpoint and os.path.exists(segments_filename):
        segments = torch.load(segments_filename).cuda()
        embeddings = torch.load(embeddings_filename).cuda()
    else:
        segments, embeddings = simple_sam.segment_LF(LF)
        torch.save(segments, segments_filename)
        torch.save(embeddings, embeddings_filename)
    mapping = get_merge_segments_mapping(segments, embeddings)
    segments = segments.cpu().numpy()
    segments = np.vectorize(lambda x: mapping.get(x, x))(segments)
    torch.save(segments, merged_filename)
    segments = post_process_segments(segments)
    visualize_segments(
        np.stack(segments).sum(axis=0),
        LF,
        filename=vis_filename,
    )
    for i, segment in enumerate(segments):
        visualize_segments(
            segment.astype(np.uint32),
            LF * (segment).astype(np.int32)[:, :, :, :, None],
            filename=f"imgs/{str(i).zfill(3)}.png",
        )
    return segments


if __name__ == "__main__":
    dir = "/home/cedaradmin/blender/lightfield/LFPlane/f00051/png"
    segments = main(dir)
