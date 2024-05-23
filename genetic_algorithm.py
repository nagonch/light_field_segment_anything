import torch


class GeneticAlg:
    def __init__(self, similarity_matrix, pop_size=10, k_tournament=3):
        self.pop_size = pop_size
        self.k_tournament = k_tournament
        self.similarity_matrix = similarity_matrix
        self.subviews, self.segments = similarity_matrix.shape

    def initial_population(self):
        population = torch.randint(
            self.segments,
            size=(
                self.pop_size,
                self.subviews,
            ),
        ).cuda()
        return population

    def fitness_function(self, population):
        row_indices = torch.arange(self.subviews).unsqueeze(0).expand(self.pop_size, -1)
        result = self.similarity_matrix[row_indices, population].mean(axis=1)
        return result

    def tournament_selection(self, population, fitness):
        parents = []
        for i in range(self.pop_size):
            inds = torch.randperm(fitness.shape[0])[: self.k_tournament]
            winner = inds[fitness[inds].argmax()]
            parents.append(population[winner])
        parents = torch.stack(parents)
        return parents

    def run(self):
        population = self.initial_population()
        fitness = self.fitness_function(population)
        parents = self.tournament_selection(population, fitness)
        print(parents)


if __name__ == "__main__":
    subviews = 10
    segments = 9
    pop_size = 10
    similarity_matrix = torch.rand(subviews, segments).cuda()
    alg = GeneticAlg(similarity_matrix, pop_size=pop_size)
    population = alg.initial_population()
    alg.run()
