import torch
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from utils import (
    shift_binary_mask,
    project_point_onto_line,
    test_mask,
    binary_mask_centroid,
    visualize_segments,
    get_subview_indices,
    test_mask,
    CONFIG,
)
from tqdm import tqdm
from utils import get_process_to_segments_dict

# import torch.multiprocessing as mp
import multiprocessing as mp

mp.set_start_method("spawn", force=True)


class LF_segment_merger:
    @torch.no_grad()
    def __init__(self, segments, embeddings):
        self.segments = segments
        self.embeddings = embeddings
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2
        self.subview_indices = get_subview_indices(self.s_size, self.t_size)
        self.epipolar_line_vectors = self.get_epipolar_line_vectors()
        self.central_cegments = self.get_central_segments()

    @torch.no_grad()
    def get_epipolar_line_vectors(self):
        epipolar_line_vectors = (
            torch.tensor([self.s_central, self.t_central]).cuda() - self.subview_indices
        ).float()
        aspect_ratio_matrix = (
            torch.diag(torch.tensor([self.v_size, self.u_size])).float().cuda()
        )  # in case the image is non-square
        epipolar_line_vectors = (aspect_ratio_matrix @ epipolar_line_vectors.T).T
        epipolar_line_vectors = F.normalize(epipolar_line_vectors)
        epipolar_line_vectors = epipolar_line_vectors.reshape(
            self.s_size, self.t_size, 2
        )
        return epipolar_line_vectors

    @torch.no_grad()
    def get_central_segments(self):
        central_segments = torch.unique(self.segments[self.s_central, self.t_central])[
            1:
        ]
        segment_sums = torch.stack(
            [(self.segments == i).sum() for i in central_segments]
        ).cuda()
        central_segments = central_segments[
            torch.argsort(segment_sums, descending=True)
        ]
        return central_segments

    @torch.no_grad()
    def calculate_peak_metric(
        self,
        mask_central,
        central_mask_centroid,
        mask_subview,
        epipolar_line_vector,
        metric=BinaryJaccardIndex().cuda(),
    ):
        epipolar_line_point = binary_mask_centroid(mask_subview)
        displacement = project_point_onto_line(
            epipolar_line_point, epipolar_line_vector, central_mask_centroid
        )
        vec = torch.round(epipolar_line_vector * displacement).long()
        mask_new = shift_binary_mask(mask_subview, vec)
        iou = metric(mask_central, mask_new).item()
        return iou

    @torch.no_grad()
    def calculate_cosine_distance(self, central_segment, subview_segment):
        central_embedding = self.embeddings[central_segment.item()][0][None]
        subview_embedding = self.embeddings[subview_segment.item()][0][None]
        return F.cosine_similarity(central_embedding, subview_embedding)[0].item()

    @torch.no_grad()
    def find_match(self, main_mask, main_mask_centroid, s, t, main_mask_segment_num):
        segments_result = []
        metrics_result = []
        for segment_num in torch.unique(self.segments[s, t])[1:]:  # TODO: parallelize
            seg = self.segments[s, t] == segment_num
            if test_mask(seg, main_mask_centroid, self.epipolar_line_vectors[s, t]):
                segments_result.append(segment_num.item())
                if CONFIG["distance-metric"] == "iou":
                    metric = self.calculate_peak_metric(
                        main_mask,
                        main_mask_centroid,
                        seg,
                        self.epipolar_line_vectors[s, t],
                    )
                elif CONFIG["distance-metric"] == "cosine":
                    metric = self.calculate_cosine_distance(
                        main_mask_segment_num, segment_num
                    )
                else:
                    raise ValueError("Config distance metric not recognized")
                metrics_result.append(metric)
        if not segments_result or np.max(metrics_result) <= CONFIG["metric-threshold"]:
            return -1  # match not found
        return segments_result[np.argmax(metrics_result)]

    @torch.no_grad()
    def find_matches(self, main_mask, main_mask_centroid, main_mask_segment_num):
        matches = []
        for s in range(self.s_size):  # TODO: parallelize
            for t in range(self.t_size):
                if s == self.s_central and t == self.t_central:
                    continue
                segment_match = self.find_match(
                    main_mask, main_mask_centroid, s, t, main_mask_segment_num
                )
                if segment_match >= 0:
                    matches.append(segment_match)
        return matches

    @torch.no_grad()
    def get_result_masks(self):
        for segment_num in tqdm(self.central_cegments):
            main_mask = (self.segments == segment_num)[self.s_central, self.t_central]
            main_mask_centroid = binary_mask_centroid(main_mask)
            matches = self.find_matches(main_mask, main_mask_centroid, segment_num)
            self.segments[torch.isin(self.segments, torch.tensor(matches).cuda())] = (
                segment_num
            )
        self.segments[
            ~torch.isin(
                self.segments,
                torch.unique(self.segments[self.s_central, self.t_central]),
            )
        ] = 0
        return self.segments


def parallelize_segments(i, results, segments, proc_to_seg_dict, embeddings):
    segments_i = (
        segments.float() * torch.isin(segments, proc_to_seg_dict[i]).float()
    ).long()
    merger = LF_segment_merger(segments_i, embeddings)
    results[i] = merger.get_result_masks().cpu()


def get_merged_segments(segments, embeddings):
    s_central, t_central = segments.shape[0] // 2, segments.shape[1] // 2
    if (
        torch.unique(segments[s_central, t_central]).shape[0]
        >= CONFIG["min-central-segments-for-parallel"]
    ):
        proc_to_seg_dict = get_process_to_segments_dict(CONFIG["embeddings-filename"])
        result_segments_list = mp.Manager().list(
            [None] * CONFIG["n-parallel-processes"]
        )
        processes = []
        for rank in range(CONFIG["n-parallel-processes"]):
            p = mp.Process(
                target=parallelize_segments,
                args=(
                    rank,
                    result_segments_list,
                    segments,
                    proc_to_seg_dict,
                    embeddings,
                ),
            )
            p.start()
            processes.append(p)
        # Wait for all processes to complete
        for p in tqdm(processes):
            p.join()
        result = torch.stack(list(result_segments_list)).sum(axis=0)
    else:
        merger = LF_segment_merger(segments, embeddings)
        result = merger.get_result_masks()
    return result


if __name__ == "__main__":
    pass
    # segments = torch.tensor(torch.load("segments.pt")).cuda()
    # print(get_merged_segments(segments))
