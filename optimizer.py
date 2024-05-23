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
        result = []
        for i, x_i in enumerate(x):
            result.append(self.similarities[i][int(x_i)])
        result = np.array(result)
        return -result.sum()

    def run(self):
        result = differential_evolution(self.loss, self.bounds, integrality=True)
        return result


if __name__ == "__main__":
    subviews = 3
    segments = 2
    sims = np.random.uniform(size=(subviews, segments))
    print(sims)
    opt = OptimizerEvolution(sims)
    print(opt.run())
