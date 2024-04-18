""" By Matthew Baas (rf5.github.io) """

import torch
import torch.nn.functional as F
import yaml

with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def kmeans_pp_init(X, k, dist_func, tol=1e-9):
    """
    `X` is (d, N) , `k` is int;
    uses kmeanspp init from https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
    """
    means = torch.empty(X.shape[0], k, dtype=X.dtype, device=X.device)
    means[:, 0] = X[:, torch.randint(0, X.shape[1], (1,))[0]]
    for i in range(1, k):
        D = dist_func(X, means[:, :i]).min(dim=-1).values  # (N, k)
        D = torch.clamp(D, tol)
        # naive way of doing this
        # probs = D / D.sum(dim=0)
        # smarter way of doing this to prevent numerical errors
        logp = D.log() - D.sum(dim=0).log()
        pmf = torch.distributions.Categorical(logits=logp, validate_args=True)
        ind = pmf.sample()
        means[:, i] = X[:, ind]
    return means


def euclid_dist(X, means):
    """`X` is (d, N), `means` is (d, K), returns dist matrix of shape (N, K)"""
    dist = ((X[..., None] - means[:, None]) ** 2).sum(dim=0)
    return dist


def cosine_dist(X, means):
    """`X` is (d, N), `means` is (d, K), returns dist matrix of shape (N, K)"""
    dist = 1 - F.cosine_similarity(X[..., None], means[:, None], dim=0)
    return dist


def classes_penalty(t_jn, classes, eps=1e-8):
    cluster_labels = t_jn.sum(axis=1)
    penalties = []
    for i in torch.unique(cluster_labels):
        cluster_labels = classes[cluster_labels == i]
        _, label_counts = torch.unique(cluster_labels, return_counts=True)
        label_probs = label_counts / len(cluster_labels)
        penalties.append(-torch.sum(label_probs * torch.log(label_probs + eps)))
    return torch.stack(penalties).cuda()


def k_means(
    X: torch.Tensor,
    classes: torch.Tensor,
    k: int,
    tol=1e-9,
    times=CONFIG["k-means-n-iterations"],
    dist="euclid",
    init="kmeanspp",
    lambda_reg=CONFIG["k-means-reg-parameter"],
    verbose=False,
):
    """
    k-means for `X` (d, N) and `k` classes, where d is vector dimension and N is number of vectors.
    Tries to fit a kmeans model `times` number of times, returning the results for the best run.
    The kmeans uses `dist` (either 'euclid' or 'cosine') for distance function.
    The kmeans uses `init` cluster initialization (either 'kmeanspp' or 'random').
    Returns (means, cluster assignments, best loss)"""
    dist_func = euclid_dist if dist == "euclid" else cosine_dist
    best_loss = torch.tensor(float("inf"), dtype=torch.float, device=X.device)
    best_means = None
    best_t_jn = None
    for t in range(times):
        if init == "kmeanspp":
            means = kmeans_pp_init(X, k, dist_func)
        else:
            means = X[:, torch.randperm(X.shape[-1], device=X.device)[:k]]  # (d, k)
        new_means = 0

        while ((new_means - means) ** 2).sum() > tol:
            # E step
            new_means = means
            dists = dist_func(X, means)
            assigned_classes = dists.argmin(dim=-1)
            del dists
            t_jn = torch.zeros((X.shape[-1], k), device=X.device)
            t_jn[torch.arange(t_jn.shape[0], device=X.device), assigned_classes] = 1
            # M step
            for i in range(k):
                class_i_samples = X[:, assigned_classes == i]
                # only update the mean if a sample is assigned to this cluster.
                if class_i_samples.shape[-1] > 0:
                    new_means[:, i] = class_i_samples.mean(dim=-1)
            # class means (d, k)
            loss = (
                t_jn[None] * dist_func(X, new_means)
            ).mean() + lambda_reg * classes_penalty(
                t_jn, classes
            ).mean()  # (d, n, k)

        if loss < best_loss:
            if verbose:
                print(f"Run {t:4d}: found new best loss: {loss:7f}")
            best_loss = loss
            best_means = new_means
            best_t_jn = t_jn
    cluster_assignments = best_t_jn.argmax(dim=-1)
    return best_means, cluster_assignments, best_loss


if __name__ == "__main__":
    pass
