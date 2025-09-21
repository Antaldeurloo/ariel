
import numpy as np
from A2_template import generate_genome, generate_initial_population, select_parents, mutate, reproduce, controller, Metadata
import datetime


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
    test_hio = np.array([np.random.randint(1,10,3) for _ in range(20)])
    test_hio[:, 1] = 3 * np.ones(20)
    test_sizes = np.random.randint(1,100, 20)
    for hio, size in zip(test_hio, test_sizes):
        config = Metadata(
            hio,
            10,
            1,
            size,
            1,
            1
        )
        population = generate_initial_population(config)
        assert len(population) == size
        for individual in population:
            assert list(individual.keys()) == ["genome", "fitness"]
            result = (int(2*(hio[0]*(hio[1]+hio[2]))),)
            assert individual["genome"].shape == result
            assert individual["fitness"] is not None
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
    p_mutate = 1
    for hio, size in zip(test_hio, test_sizes):
        population = generate_initial_population(size, hio)
        for individual in population:
            genotype = individual['genome']
            mut_genotype = mutate(genotype, p_mutate)
            assert isinstance(mut_genotype, np.ndarray)
            assert genotype.shape == mut_genotype.shape
            assert not (genotype == mut_genotype).all()
    pass


def reproduce_test():
    hio = (2,2,2)
    test_sizes = np.random.randint(6,100,20)
    for size in test_sizes:
        population = generate_initial_population(size, hio)
        fitness = np.arange(size)
        for individual, ind_fit in zip(population, fitness):
            individual["fitness"] = ind_fit
        parents, non_parents = select_parents(population)
        new_pop = reproduce(parents, non_parents, 1, size)
        assert len(new_pop) == size
        for individual in new_pop:
            assert list(individual.keys()) == ["genome", "fitness"]
            result = (int(2*(hio[0]*(hio[1]+hio[2]))),)
            assert individual["genome"].shape == result
        for ind1, ind2 in zip(population, new_pop):
            differences = []
            differences.append((ind1['genome'] == ind2['genome']).all())
        assert not all(differences)
    pass

# class Data:
#     def __init__(self):
#         self.ctrl = np.zeros(8)

# def controller_test(data):
#     hio = (12, 8, 8)
#     genome = generate_genome(hio)
#     controller(0, data, 0, genome, hio)
#     print(data.ctrl)

#     pass


def main():
    generate_individual_test()
    # generate_population_test()
    # parent_selection_test()
    # mutate_test()
    # reproduce_test()
    print("all tests passed")


if __name__ == "__main__":
    main()
