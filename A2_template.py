# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from typing import Literal, cast
import random
import datetime

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.ec.a005 import Crossover
from ariel.ec.a000 import IntegerMutator

from ariel.utils.runners import simple_runner

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []


class Metadata:
    def __init__(self, dof, duration, n_generations, size, p_mut, offspring_multiplier):
        self.dof = dof
        self.training_duration = duration
        self.n_generations = n_generations
        self.population_size = size
        self.p_mut = p_mut
        self.offspring_multiplier = offspring_multiplier
        self.num_evals = 0


def train_model(config: Metadata):
    """Training model is done here"""
    with open("results.csv", "w", encoding="utf-8") as f:
        f.write(f"Starting evolution with dof = {config.dof}\n")

    population = generate_initial_population(config)
    fitness_history = []

    for i in range(config.n_generations):
        # Basically main() but for each generation
        parents, non_parents = select_parents(population)
        population = reproduce(parents, population, config)
        fitness_list = np.array([ind["fitness"] for ind in population])
        print(f'Generation {i}: hi: {np.max(fitness_list)}, lo: {np.min(fitness_list)}, mean: {np.mean(fitness_list)}')
        fitness_list = np.array([ind["fitness"] for ind in population])
        fitness_history.append(fitness_list)
    return population[np.argmax(fitness_list)]["genome"], np.array(fitness_history)


def generate_initial_population(config: Metadata):
    """Generate initial population of random genomes"""

    population = []
    for _ in range(config.population_size):
        genome = generate_genome(config.dof)
        population.append({"genome": genome, "fitness": None})
    for ind in population:
        ind["fitness"] = run_individual_trial(
            ind['genome'],
            config
            )

    return population


def select_parents(population):
    """Select parents for the next generation through tournament selection"""
    winners = []
    losers = []
    random.shuffle(population)
    for idx in range(0, len(population)-1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        if ind_i["fitness"] > ind_j["fitness"]:
            winners.append(ind_i)
            losers.append(ind_j)
        else:
            winners.append(ind_j)
            losers.append(ind_i)
    return winners, losers


def reproduce(parents, population, config: Metadata):
    """Reproduce new individuals from parents"""

    for idx in range(0, len(parents)-1, 2):
        parent_i_genotype = parents[idx]["genome"]
        parent_j_genotype = parents[idx+1]["genome"]

        # crossover
        children = []
        for i in range(config.offspring_multiplier):
            genotype_i, genotype_j = Crossover.one_point(
                cast("list[float]", parent_i_genotype),
                cast("list[float]", parent_j_genotype)
            )

            mut_genotype_i = mutate(np.array(genotype_i), config.p_mut)
            mut_genotype_j = mutate(np.array(genotype_j), config.p_mut)
            child_i = {"genome": mut_genotype_i,
                       "fitness": run_individual_trial(
                            mut_genotype_i,
                            config
                        )}
            child_j = {"genome": mut_genotype_j,
                       "fitness": run_individual_trial(
                            mut_genotype_j,
                            config
                        )}
            children.extend([child_i, child_j])
        population.extend(children)
    fitness_list = np.array([ind["fitness"] for ind in population])
    sorted_indices = np.argsort(fitness_list)[-config.population_size:]

    return [population[i] for i in sorted_indices]


def mutate(genotype, p_mutate):
    mut_genotype = genotype.copy()
    mut_weights = IntegerMutator.float_creep(
        individual=cast("list[float]", mut_genotype),
        span=np.pi/4,
        mutation_probability=p_mutate
    )
    return np.array(mut_weights)


def fitness_function(movement_history) -> float:
    """The further from the center the better"""

    # return np.linalg.norm(movement_history[-1] - movement_history[0])
    return -1 * (movement_history[-1][1] - movement_history[0][1]) - abs(movement_history[-1][0] - movement_history[0][0])


def run_individual_trial(genome, config: Metadata) -> float:
    """Run a single trial of the genome"""
    config.num_evals += 1
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    # No need to change anything above this line (i think)

    HISTORY.clear()
    mujoco.set_mjcb_control(
        lambda m, d: controller(m, d, to_track, genome, config.dof)
        )

    # Running the simulation without viewer
    simple_runner(
        model=model,
        data=data,
        duration=config.training_duration
    )

    # print("Fitness:", fitness_function(HISTORY))

    return fitness_function(HISTORY)


def generate_genome(dof):
    genome = np.array([np.random.randn() for _ in range(dof)])
    return genome


def oscillator(t, weights):
    frequencies = weights[:8]
    amplitudes = weights[8:16]
    offsets_y = weights[16:24]
    offsets_x = weights[24:]

    return amplitudes * np.sin(2 * np.pi * frequencies * (t - offsets_x)) + offsets_y


def controller(model, data, to_track, genome, dof) -> None:
    """Function to make the model move"""
    delta = oscillator(data.time, genome)
    data.ctrl = delta
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
    HISTORY.append(to_track[0].xpos.copy())
    return None


def show_qpos_history(history: list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')

    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show()


def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None) # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()     # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # No need to change anything above this line in the main (i think)
    config = Metadata(
        dof=32,
        duration=30.0,
        n_generations=30,
        size=26,
        p_mut=0.5,
        offspring_multiplier=2
        )

    HISTORY.clear()  # I don't know why I re-use HISTORY, but whatever
    best_genome, fitness_hist = train_model(config)
    np.savetxt('fitness_hist', fitness_hist)
    np.savetxt('genome', best_genome)
    std = np.std(fitness_hist, axis=0)
    print(best_genome)
    # Set the control callback function
    # This is called every time step to get the next action. 
    # Probably something like this to use our controller 
    # best_genome = generate_genome((12,8,8))
    mujoco.set_mjcb_control(
        lambda m, d: controller(m, d, to_track, best_genome, config.dof)
        )

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    # viewer.launch(
    #    model=model,  # type: ignore
    #    data=data,
    # )
    print(f'ended after {config.num_evals} fitness evaluations')

    # print("Fitness:", fitness_function(HISTORY))

    # show_qpos_history(HISTORY)
    mean = np.mean(fitness_hist, axis=0)
    upper = mean + std
    lower = mean - std
    plt.plot(np.arange(len(mean)), mean, '-b', label='mean')
    plt.fill_between(np.arange(len(mean)), upper, lower, alpha=0.5, color='b')
    plt.plot(np.arange(len(mean)), np.max(fitness_hist, axis=0), '-k', label='max fitness')
    plt.legend()

    plt.show()
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    PATH_TO_VIDEO_FOLDER = "./__videos__"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    video_renderer(
        model,
        data,
        duration=30,
        video_recorder=video_recorder,
    )


if __name__ == "__main__":
    main()