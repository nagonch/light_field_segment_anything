import numpy as np
from scipy.optimize import differential_evolution


class OptimizerEvolution:
    def __init__(self, similarities):
        self.subviews, self.segments = similarities.shape
        self.similarities = similarities
        self.bounds = [
            [0, self.segments - 1],
        ] * self.subviews

    def loss(self, x):
        result = self.similarities[np.arange(x.shape[0]), x.astype(np.int32)]
        return -result.sum() - (x % 2).sum()

    def run(self):
        result = differential_evolution(self.loss, self.bounds, integrality=True)
        return result.x


if __name__ == "__main__":
    subviews = 5
    segments = 4
    sims = np.random.uniform(size=(subviews, segments))
    opt = OptimizerEvolution(sims)
    result = opt.run()
    print(sims)
    print(result)
    print(opt.loss(result))
    print(opt.loss(sims.argmax(axis=1)))
