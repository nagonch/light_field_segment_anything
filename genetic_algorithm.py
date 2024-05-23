import torch


class GeneticAlg:
    def __init__(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix
        self.subviews, self.segments = similarity_matrix.shape

    def initial_population(self):
        population = torch.randint(
            segments,
            size=(subviews,),
        ).cuda()
        return population

    def fitness_function(self, population):
        index_vector = population.unsqueeze(1)
        result = torch.gather(self.similarity_matrix, 1, index_vector).squeeze(1)
        return result


if __name__ == "__main__":
    subviews = 4
    segments = 3
    similarity_matrix = torch.rand(subviews, segments).cuda()
    alg = GeneticAlg(similarity_matrix)
    population = alg.initial_population()
    print(alg.fitness_function(population))
