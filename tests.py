
import numpy as np
from A2_template import generate_genome, generate_initial_population, select_parents, mutate, reproduce


#TODO
"""
generate_individual
parent_selection
reproduce
mutate
"""


def generate_individual_test():
    test_values = [np.random.randint(1,10,3) for _ in range(20)]
    for hio in test_values:
        genome = generate_genome(hio)
        result = (int(2*(hio[0]*(hio[1]+hio[2]))),)
        assert genome.shape == result
        assert genome.dtype == "float64"
        assert isinstance(genome, np.ndarray)
    pass


def generate_population_test():
    test_hio = [np.random.randint(1,10,3) for _ in range(20)]
    test_sizes = np.random.randint(1,100, 20)
    for hio, size in zip(test_hio, test_sizes):
        population = generate_initial_population(size, hio)
        assert len(population) == size
        for individual in population:
            assert list(individual.keys()) == ["genome", "fitness"]
            result = (int(2*(hio[0]*(hio[1]+hio[2]))),)
            assert individual["genome"].shape == result
            assert individual["fitness"] is None
    pass


def parent_selection_test():
    hio = (5, 12, 5)
    test_sizes = np.random.randint(1,100,20)
    for size in test_sizes:
        population = generate_initial_population(size, hio)
        fitness = np.arange(size)
        for individual, ind_fit in zip(population, fitness):
            individual["fitness"] = ind_fit
        parents, non_parents = select_parents(population)
        assert len(parents) == size // 2
        for individual in parents:
            assert list(individual.keys()) == ["genome", "fitness"]
            result = (int(2*(hio[0]*(hio[1]+hio[2]))),)
            assert individual["genome"].shape == result
            assert isinstance(individual["genome"], np.ndarray)
            assert individual["fitness"] is not None

    pass


def mutate_test():
    test_hio = [np.random.randint(1,10,3) for _ in range(20)]
    test_sizes = np.random.randint(1,100, 20)
    p_mutate = 0.5
    for hio, size in zip(test_hio, test_sizes):
        population = generate_initial_population(size, hio)
        for individual in population:
            genotype = individual['genome']
            mut_genotype = mutate(genotype, p_mutate)
            assert isinstance(mut_genotype, np.ndarray)
            assert genotype.shape == mut_genotype.shape
            assert genotype != mut_genotype
    pass


def reproduce_test():
    pass


def main():
    generate_individual_test()
    generate_population_test()
    parent_selection_test()
    mutate_test()
    reproduce_test()
    print("all tests passed")


if __name__ == "__main__":
    main()
