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
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


class LF_segment_merger:
    @torch.no_grad()
    def __init__(self, segments, embeddings_filename="embeddings.pt"):
        self.segments = segments
        self.embeddings_filename = embeddings_filename
        self.s_size, self.t_size, self.u_size, self.v_size = segments.shape
        self.s_central, self.t_central = self.s_size // 2, self.t_size // 2
        self.subview_indices = get_subview_indices(self.s_size, self.t_size)
        self.epipolar_line_vectors = self.get_epipolar_line_vectors()

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
    def get_central_segments(self, segments):
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
    def find_match(self, main_mask, main_mask_centroid, s, t):
        segments_result = []
        max_ious_result = []
        for segment_num in torch.unique(self.segments[s, t])[1:]:  # TODO: parallelize
            seg = self.segments[s, t] == segment_num
            if test_mask(seg, main_mask_centroid, self.epipolar_line_vectors[s, t]):
                segments_result.append(segment_num.item())
                max_iou = self.calculate_peak_metric(
                    main_mask,
                    main_mask_centroid,
                    seg,
                    self.epipolar_line_vectors[s, t],
                )
                max_ious_result.append(max_iou)
        if not segments_result or np.max(max_ious_result) <= CONFIG["metric-threshold"]:
            return -1  # match not found
        return segments_result[np.argmax(max_ious_result)]

    @torch.no_grad()
    def find_matches(self, main_mask, main_mask_centroid):
        matches = []
        for s in range(self.s_size):  # TODO: parallelize
            for t in range(self.t_size):
                if s == self.s_central and t == self.t_central:
                    continue
                segment_match = self.find_match(main_mask, main_mask_centroid, s, t)
                if segment_match >= 0:
                    matches.append(segment_match)
        return matches

    @torch.no_grad()
    def get_result_masks_i(self, i, results, process_to_segments_dict):
        segments_i = (
            self.segments.float()
            * torch.isin(self.segments, process_to_segments_dict[i]).float()
        ).long()
        for segment_num in tqdm(self.get_central_segments(segments_i)):
            main_mask = (segments_i == segment_num)[self.s_central, self.t_central]
            main_mask_centroid = binary_mask_centroid(main_mask)
            matches = self.find_matches(main_mask, main_mask_centroid)
            segments_i[torch.isin(segments_i, torch.tensor(matches).cuda())] = (
                segment_num
            )
        segments_i[
            ~torch.isin(
                segments_i,
                torch.unique(segments_i[self.s_central, self.t_central]),
            )
        ] = 0
        results[i] = segments_i

    @torch.no_grad()
    def get_result_masks_parallel(self):
        process_to_segments_dict = get_process_to_segments_dict(
            self.embeddings_filename
        )
        result_segments = mp.Manager().list([None] * CONFIG["n-parallel-processes"])
        # Spawn processes
        processes = []
        for rank in range(CONFIG["n-parallel-processes"]):
            p = mp.Process(
                target=self.get_result_masks_i,
                args=(rank, result_segments, process_to_segments_dict),
            )
            processes.append(p)
            p.start()
        # Wait for all processes to complete
        for p in processes:
            p.join()
        print(result_segments)
        raise
        return result_segments

    @torch.no_grad()
    def get_result_masks(self):
        for segment_num in tqdm(self.get_central_segments(self.segments)):
            main_mask = (self.segments == segment_num)[self.s_central, self.t_central]
            main_mask_centroid = binary_mask_centroid(main_mask)
            matches = self.find_matches(main_mask, main_mask_centroid)
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


if __name__ == "__main__":
    # mask = torch.zeros((256, 341))
    # mask = draw_line_in_mask(mask, (0, 0), (230, 240))
    # plt.imshow(mask)
    # plt.show()
    # plt.close()
    # raise
    from random import randint
    from utils import get_process_to_segments_dict

    # process_to_segments_dict = get_process_to_segments_dict("embeddings.pt")
    segments = torch.tensor(torch.load("segments.pt")).cuda()
    # result_segments = torch.zeros_like(segments).cuda()
    # for i in range(CONFIG["n-parallel-processes"]):
    #     segments_i = (
    #         segments.float() * torch.isin(segments, process_to_segments_dict[i]).float()
    #     ).long()
    #     merger = LF_segment_merger(segments_i)
    #     result_masks = merger.get_result_masks()
    #     result_segments += result_masks
    # torch.save(result_segments, "segments_merged.pt")
    merger = LF_segment_merger(segments)
    merger.get_result_masks()
    # segment_merger = LF_segment_merger(segments)
    # print(segment_merger)
    # find_matches_RANSAC(segments, 331232, n_data_points=70)
    # print(get_result_masks(segments))
    # central_test_segment = 32862
    # print(get_segmentation(segments, central_test_segment))
