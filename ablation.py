import torch
from tqdm import tqdm
from torchmetrics.classification import BinaryJaccardIndex
import networkx as nx
import torch.nn.functional as F
from utils import CONFIG


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


# Naive version of the algorithm using graphs
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
                    metric_val = calculate_segments_metric(segments, embeddings, i, j)
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


if __name__ == "__main__":
    pass
