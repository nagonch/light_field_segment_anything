import numpy as np
from scipy.optimize import differential_evolution
from utils import masks_iou
import torch


class OptimizerEvolution:
    def __init__(
        self,
        similarities,
        segment_matrix,
        central_segment,
        lambda_reg=1e-3,
        maxiter=1000,
    ):
        self.maxiter = maxiter
        self.segment_matrix = segment_matrix
        self.central_segment = central_segment
        self.lambda_reg = lambda_reg
        self.subviews, self.segments = similarities.shape
        self.similarities = similarities
        self.bounds = [
            [0, self.segments - 1],
        ] * self.subviews

    def loss(self, x):
        result = self.similarities[
            torch.arange(x.shape[0]).cuda(), torch.tensor(x).cuda().long()
        ]
        result_value = (-result.sum()).item()
        return result_value

    def run(self):
        result = differential_evolution(
            self.loss, self.bounds, integrality=True, maxiter=self.maxiter
        )
        return torch.tensor(result.x).cuda().long()


if __name__ == "__main__":
    sim_matrix = torch.load("sim_matrix.pt")
    segment_matrix = torch.load("segment_matrix.pt")
    segment_indices = torch.load("segment_indices.pt")
    central_mask = torch.load("central_mask.pt")
    opt = OptimizerEvolution(sim_matrix, segment_matrix, central_mask)
    result = opt.run()
    print(result)
    print(sim_matrix[torch.arange(sim_matrix.shape[0]), result])
    print(sim_matrix.max(axis=1)[0])
    print(sim_matrix[torch.arange(sim_matrix.shape[0]), result].mean())
