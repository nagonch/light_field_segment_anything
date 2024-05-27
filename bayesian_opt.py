import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf_discrete
from utils import masks_cross_ssim


class OptimizerBayes:
    def __init__(
        self,
        similarities,
        segment_matrix,
        central_segment,
        N_train_points=1000,
        search_space_size=1000,
        lambda_reg=10,
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
        subviews_index = torch.arange(self.n_subviews).cuda()
        train_X = torch.randint(
            0,
            self.n_segments,
            size=(
                self.N_train_points,
                self.n_subviews,
            ),
        ).cuda()
        train_X[0] = torch.zeros((self.n_subviews,)).cuda()
        ssims = torch.zeros((train_X.shape[0])).cuda()
        for vector_num, train_vector in enumerate(train_X):
            segments = self.segment_matrix[subviews_index, train_vector, :, :]
            segments = torch.cat((segments, self.central_segment[None]), dim=0)
            ssims[vector_num] = masks_cross_ssim(segments)
        train_Y = (
            self.similarities[subviews_index, train_X]
            .mean(axis=-1)
            .double()
            .cuda()[None]
        )
        train_Y = (train_Y + self.lambda_reg * ssims).T
        choices = torch.randint(
            0,
            self.n_segments,
            size=(
                self.search_space_size,
                self.n_subviews,
            ),
        ).cuda()
        choices[0] = torch.zeros((self.n_subviews)).cuda()
        train_X = train_X.double()
        return train_X, train_Y, choices

    def maxinimize(self):
        train_X, train_Y, choices = self.prepare_data()
        train_Y = standardize(train_Y)
        gp = SingleTaskGP(train_X, train_Y)

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        UCB = UpperConfidenceBound(gp, beta=0.1)
        candidate, _ = optimize_acqf_discrete(
            UCB,
            choices=choices,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        return candidate


if __name__ == "__main__":
    sim_matrix = torch.load("sim_matrix.pt")[:, :5].cuda()
    segment_matrix = torch.load("segment_matrix.pt")[:, :5, :, :].cuda()
    segment_indices = torch.load("segment_indices.pt").cuda()
    central_mask = torch.load("central_mask.pt").cuda()
    opt = OptimizerBayes(sim_matrix, segment_matrix, central_mask)
    print(opt.maxinimize())
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
