import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf_discrete
from matplotlib import pyplot as plt

N = 1000
N_choices = 1000

sim_matrix = torch.load("sim_matrix.pt")[:, :5]
m, n = sim_matrix.shape
train_X = torch.randint(
    0,
    n,
    size=(
        N,
        m,
    ),
)
choices = torch.randint(
    0,
    20,
    size=(
        N_choices,
        81,
    ),
)
train_Y = sim_matrix[torch.arange(m), train_X].sum(axis=-1).double().cuda()[None].T
train_X = train_X.double().cuda()
train_Y = standardize(train_Y)

gp = SingleTaskGP(train_X, train_Y)

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

UCB = UpperConfidenceBound(gp, beta=0.1)
bounds = torch.stack([torch.zeros(81), 19 * torch.ones(81)])
candidate, acq_value = optimize_acqf_discrete(
    UCB,
    choices=choices,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)
print(candidate)
