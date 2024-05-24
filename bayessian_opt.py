import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf_discrete
from matplotlib import pyplot as plt
from itertools import combinations


class OptimizerBayes:
    def __init__(
        self,
        similarities,
        segment_matrix,
        central_segment,
        N_train_points=1000,
        search_space_size=1000,
        lambda_reg=1e-3,
    ):
        self.segment_matrix = segment_matrix
        self.central_segment = central_segment
        self.lambda_reg = lambda_reg
        self.subviews, self.segments = similarities.shape
        self.similarities = similarities
        self.N_train_points = N_train_points
        self.search_space_size = search_space_size
        self.n_subviews, self.n_segments = similarities.shape
        self.N_train_points = N_train_points
        self.search_space_size = search_space_size

    def prepare_data(self):
        train_X = torch.randint(
            0,
            self.n_segments,
            size=(
                self.N_train_points,
                self.n_subviews,
            ),
        ).cuda()
        train_X[0] = torch.zeros((self.n_subviews,)).cuda()
        choices = torch.randint(
            0,
            self.n_segments,
            size=(
                self.search_space_size,
                self.n_subviews,
            ),
        ).cuda()
        choices[0] = torch.zeros((self.n_subviews)).cuda()
        print(train_X, choices)


if __name__ == "__main__":
    sim_matrix = torch.load("sim_matrix.pt").cuda()
    segment_matrix = torch.load("segment_matrix.pt").cuda()
    segment_indices = torch.load("segment_indices.pt").cuda()
    central_mask = torch.load("central_mask.pt").cuda()
    opt = OptimizerBayes(sim_matrix, segment_matrix, central_mask)
    opt.prepare_data()
    # N = 1000
    # N_choices = 2000

    # sim_matrix = torch.load("sim_matrix.pt")[:, :3].cuda()
    # m, n = sim_matrix.shape
    # train_X = torch.randint(
    #     0,
    #     n,
    #     size=(
    #         N,
    #         m,
    #     ),
    # ).cuda()

    # train_X[0] = torch.zeros((1, m)).cuda()

    # choices = torch.randint(
    #     0,
    #     n,
    #     size=(
    #         N_choices,
    #         m,
    #     ),
    # ).cuda()
    # choices[0] = torch.zeros((1, m)).cuda()

    # train_Y = sim_matrix[torch.arange(m), train_X].sum(axis=-1).double().cuda()[None].T
    # train_X = train_X.double().cuda()
    # train_Y = standardize(train_Y)

    # gp = SingleTaskGP(train_X, train_Y)

    # mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    # fit_gpytorch_mll(mll)

    # UCB = UpperConfidenceBound(gp, beta=0.1)
    # candidate, acq_value = optimize_acqf_discrete(
    #     UCB,
    #     choices=choices,
    #     q=1,
    #     num_restarts=5,
    #     raw_samples=20,
    # )
    # print(candidate, sim_matrix[torch.arange(m), candidate].sum())
    # X_opt = torch.argmax(sim_matrix, axis=-1)
    # print(X_opt, sim_matrix[torch.arange(m), X_opt].sum())
