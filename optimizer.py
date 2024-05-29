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
            ]
            masks = torch.stack(masks)
            result += self.lambda_reg * masks_regularization_score(masks)
        return result

    def run(self):
        print(self.loss(0, 0, [[1, 1], [2, 2]]))
        return None


if __name__ == "__main__":
    sim_matrix = torch.load("sim_matrix.pt")
    segment_matrix = torch.load("segment_matrix.pt")
    segment_indices = torch.load("segment_indices.pt")
    central_mask = torch.load("central_mask.pt")
    opt = GreedyOptimizer(sim_matrix, segment_matrix, central_mask, segment_indices)
    result = opt.run()
    print(result)
