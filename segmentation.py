import numpy as np

import torch
from tqdm import tqdm
import yaml
import os
import imgviz
from utils import visualize_segments, get_LF, CONFIG
from sam_functions import get_sam, segment_LF
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex
import networkx as nx


def get_segments_metric(segments, i, j, metric=BinaryJaccardIndex().cuda()):
    mask_i = (segments == i).to(torch.int32)
    mask_j = (segments == j).to(torch.int32)
    segments_i = mask_i.sum(axis=(0, 1))
    segments_j = mask_j.sum(axis=(0, 1))
    metric_val = metric(segments_i, segments_j)
    return metric_val.item()


def get_merge_segments_mapping(segments):
    metric = BinaryJaccardIndex().cuda()
    s, t = segments.shape[0], segments.shape[1]
    central_inds = torch.unique(segments[s // 2][t // 2].reshape(-1))
    graphs = []
    mapping = {}
    for s_ in range(s):
        for t_ in range(t):
            if s_ == s // 2 and t_ == t // 2:
                continue
            graph = nx.Graph()
            segments_st = torch.unique(segments[s_][t_].reshape(-1))
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
                    metric_val = get_segments_metric(segments, i, j, metric)
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


def visualize_segments(segments, LF, st_border=None, filename=None):
    if st_border:
        LF = LF[st_border:-st_border, st_border:-st_border]
        segments = segments[st_border:-st_border, st_border:-st_border]
    s, t, u, v, c = LF.shape
    segments = np.transpose(segments, (0, 2, 1, 3)).reshape(s * u, t * v)
    LF = np.transpose(LF, (0, 2, 1, 3, 4)).reshape(s * u, t * v, c)
    vis = imgviz.label2rgb(
        label=segments,
        image=LF,
        colormap=imgviz.label_colormap(segments.max() + 1),
    )
    plt.imshow(vis)
    if filename:
        plt.savefig(filename)
    plt.close()


def main(
    LF_dir,
    segments_filename=CONFIG["segments-filename"],
    merged_filename=CONFIG["merged-filename"],
    segments_checkpoint=CONFIG["sam-segments-checkpoint"],
    vis_filename=CONFIG["vis-filename"],
):
    LF = get_LF(LF_dir)[7:-7, 7:-7]
    simple_sam = get_sam()
    if segments_checkpoint and os.path.exists(segments_filename):
        segments = torch.load(segments_filename).cuda()
    else:
        segments, _ = segment_LF(simple_sam, LF)
        torch.save(segments, segments_filename)
    mapping = get_merge_segments_mapping(segments)
    segments = segments.cpu().numpy()
    segments = np.vectorize(lambda x: mapping.get(x, x))(segments)
    torch.save(segments, merged_filename)
    visualize_segments(
        segments,
        LF,
        filename=vis_filename,
    )
    return segments


if __name__ == "__main__":
    dir = "/home/cedaradmin/data/lf_nonun/LFPlane/f00032/png"
    segments = main(dir)
