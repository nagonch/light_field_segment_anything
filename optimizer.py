from utils import masks_regularization_score
import torch


class GreedyOptimizer:
    def __init__(
        self,
        similarities,
        segment_matrix,
        central_segment,
        segment_indices,
        lambda_reg=1e-3,
    ):
        self.segment_indices = segment_indices
        self.segment_matrix = segment_matrix
        self.central_segment = central_segment
        self.lambda_reg = lambda_reg
        self.n_subviews, self.n_segments = similarities.shape
        self.similarities = similarities

    def loss(self, x):
        result = self.similarities[
            torch.arange(x.shape[0]).cuda(), torch.tensor(x).cuda().long()
        ]
        result_value = (-result.sum()).item()
        return result_value

    def run(self):
        return None


if __name__ == "__main__":
    sim_matrix = torch.load("sim_matrix.pt")
    segment_matrix = torch.load("segment_matrix.pt")
    segment_indices = torch.load("segment_indices.pt")
    central_mask = torch.load("central_mask.pt")
    opt = GreedyOptimizer(sim_matrix, segment_matrix, central_mask, segment_indices)
    result = opt.run()
    print(result)
    print(sim_matrix[torch.arange(sim_matrix.shape[0]), result])
    print(sim_matrix.max(axis=1)[0])
    print(sim_matrix[torch.arange(sim_matrix.shape[0]), result].mean())
