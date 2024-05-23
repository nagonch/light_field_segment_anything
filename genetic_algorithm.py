import torch


class GeneticAlg:
    def __init__(self, similarity_matrix, init_size=10):
        self.init_size = init_size
        self.similarity_matrix = similarity_matrix
        self.subviews, self.segments = similarity_matrix.shape

    def initial_population(self):
        population = torch.randint(
            segments,
            size=(
                self.init_size,
                subviews,
            ),
        ).cuda()
        return population


if __name__ == "__main__":
    subviews = 81
    segments = 20
    distance_matrix = torch.rand(subviews, segments).cuda()
    alg = GeneticAlg(distance_matrix)
    print(alg.initial_population())
