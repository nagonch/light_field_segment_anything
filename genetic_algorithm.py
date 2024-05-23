import torch


class GeneticAlg:
    def __init__(self, similarity_matrix, init_size=10):
        self.init_size = init_size
        self.similarity_matrix = similarity_matrix
        self.subviews, self.segments = similarity_matrix.shape

    def initial_population(self):
        population = torch.randint(
            self.segments,
            size=(
                self.init_size,
                self.subviews,
            ),
        ).cuda()
        return population

    def fitness_function(self, population):
        row_indices = (
            torch.arange(self.subviews).unsqueeze(0).expand(self.init_size, -1)
        )
        result = self.similarity_matrix[row_indices, population].mean(axis=1)
        return result

    def run(self):
        population = self.initial_population()
        fitness = self.fitness_function(population)
        order = torch.argsort(fitness)
        print(order)


if __name__ == "__main__":
    subviews = 10
    segments = 9
    init_size = 10
    similarity_matrix = torch.rand(subviews, segments).cuda()
    alg = GeneticAlg(similarity_matrix, init_size=init_size)
    population = alg.initial_population()
    alg.run()
