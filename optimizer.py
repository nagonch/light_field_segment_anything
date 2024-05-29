from utils import masks_regularization_score
import torch


class GreedyOptimizer:
    def __init__(
        self,
        similarities,
        segment_matrix,
        central_segment,
        segment_indices,
        lambda_reg=1e-1,
    ):
        self.segment_indices = segment_indices
        self.segment_matrix = segment_matrix
        self.central_segment = central_segment
        self.lambda_reg = lambda_reg
        self.n_subviews, self.n_segments = similarities.shape
        self.similarities = similarities

    def loss(self, subview_ind, segment_ind, chosen_segment_inds=None):
        result = self.similarities[subview_ind, segment_ind]
        if chosen_segment_inds:
            masks = [
                self.segment_matrix[subview, segment]
                for subview, segment in chosen_segment_inds
            ]
            masks += [
                self.segment_matrix[subview_ind, segment_ind],
                self.central_segment,
            ]
            masks = torch.stack(masks)
            result += self.lambda_reg * masks_regularization_score(masks)
        return result

    def run(self):
        matches = []
        chosen_segment_inds = []
        for i in range(self.n_subviews):
            print(i)
            ind_num, segment_num = torch.where(sim_matrix == sim_matrix.max())
            sim_matrix[ind_num] = torch.ones_like(sim_matrix[ind_num]) * (-torch.inf)
            matches.append(self.segment_indices[ind_num[0], segment_num[0]])
            chosen_segment_inds.append([ind_num[0], segment_num[0]])
            if i < self.n_subviews - 1:
                for candidate_i in range(self.n_subviews):
                    for candidate_j in range(self.n_segments):
                        if self.segment_indices[candidate_i, candidate_j] in matches:
                            continue
                        self.similarities[candidate_i, candidate_j] = self.loss(
                            candidate_i,
                            candidate_j,
                            chosen_segment_inds=chosen_segment_inds,
                        )
        return matches


if __name__ == "__main__":
    sim_matrix = torch.load("sim_matrix.pt")[:, :5]
    segment_matrix = torch.load("segment_matrix.pt")
    segment_indices = torch.load("segment_indices.pt")
    central_mask = torch.load("central_mask.pt")
    opt = GreedyOptimizer(sim_matrix, segment_matrix, central_mask, segment_indices)
    result = opt.run()
    print(result)
