import numpy as np

import torch
from tqdm import tqdm
import yaml
import os
import imgviz
from utils import visualize_segments, CONFIG, save_LF_image
from data import get_LF
from sam_functions import get_sam
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex
import networkx as nx
from LF_functions import calculate_peak_metric
import torch.nn.functional as F


def calculate_segments_metric(segments, embeddings, i, j):
    mask_i = (segments == i).to(torch.int32)
    mask_j = (segments == j).to(torch.int32)
    if CONFIG["edge-metric"] in ["cosine", "combination"]:
        embs_i = (embeddings * (segments == i)[:, :, :, :, None].to(torch.int32)).mean(
            axis=(0, 1, 2, 3)
        )
        embs_j = (embeddings * (segments == j)[:, :, :, :, None].to(torch.int32)).mean(
            axis=(0, 1, 2, 3)
        )
        cosine_sim = F.cosine_similarity(embs_i[None], embs_j[None])[0].item()
        if CONFIG["edge-metric"] == "cosine":
            return cosine_sim
    if CONFIG["edge-metric"] in ["iou", "combination"]:
        iou_metric = BinaryJaccardIndex().cuda()
        segments_i = mask_i.sum(axis=(0, 1))
        segments_j = mask_j.sum(axis=(0, 1))
        iou = iou_metric(segments_i, segments_j).item()
        if CONFIG["edge-metric"] == "iou":
            return iou
    return CONFIG["cosine-weight"] * cosine_sim + (1 - CONFIG["cosine-weight"]) * iou


def get_merge_segments_mapping(segments, embeddings):
    s, t = segments.shape[0], segments.shape[1]
    central_inds = torch.unique(segments[s // 2][t // 2].reshape(-1))[1:]
    graphs = []
    mapping = {}
    for s_ in tqdm(range(s), leave=False):
        for t_ in tqdm(range(t), leave=False):
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
            for i in tqdm(central_inds, leave=False):
                for j in segments_st:
                    metric_val = calculate_peak_metric(segments, i, j)
                    if metric_val < CONFIG["metric-threshold"]:
                        metric_val = 0.0
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
    LF = dataset[2][2:-2, 2:-2].detach().cpu().numpy()
    save_LF_image(np.array(LF), "input_LF.png")
    # LF = get_LF(LF_dir)
    # LF = loadmat("lego_128.mat")["LF"].astype(np.int32)[1:-1, 1:-1]
    simple_sam = get_sam()
    if segments_checkpoint and os.path.exists(segments_filename):
        segments = torch.load(segments_filename).cuda()
        embeddings = torch.load(embeddings_filename).cuda()
    else:
        segments, embeddings = simple_sam.segment_LF(LF)
        torch.save(segments, segments_filename)
        torch.save(embeddings, embeddings_filename)
    if merged_checkpoint and os.path.exists(merged_filename):
        segments = torch.load(merged_filename)
    else:
        mapping = get_merge_segments_mapping(segments, embeddings)
        segments = segments.cpu().numpy()
        segments = np.vectorize(lambda x: mapping.get(x, x))(segments)
        torch.save(segments, merged_filename)
    visualize_segments(
        segments,
        filename=vis_filename,
    )
    segments = post_process_segments(segments)
    for i, segment in enumerate(segments):
        visualize_segments(
            segment.astype(np.uint32),
            filename=f"imgs/{str(i).zfill(3)}.png",
        )
    return segments


if __name__ == "__main__":
    dir = "/home/cedaradmin/blender/lightfield/LFPlane/f00051/png"
    segments = main(dir)
