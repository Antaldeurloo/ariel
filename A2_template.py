# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from typing import Literal, cast
import random

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



#TODO optimize evaluation
def train_model(number_of_generations=1, population_size=5, training_duration=10, hio=(12,8,8)):
    """Training model is done here"""

    population = generate_initial_population(population_size, hio)

    for _ in range(number_of_generations):
        # Basically main() but for each generation

        for individual in population:
            # Evaluate each individual in the population
            if individual["fitness"] is None:
                individual["fitness"] =  run_individual_trial(individual["genome"], hio, duration=training_duration)
        parents, non_parents = select_parents(population)
        population = reproduce(parents, non_parents, p_mutate, population_size)

    
    # Return best individual
    pass

def generate_initial_population(population_size, hio):
    """Generate initial population of random genomes"""

    population = []
    for _ in range(population_size):
        genome = generate_genome(hio)

        population.append({"genome": genome, "fitness": None})

    return population

#TODO
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

#TODO
def reproduce(parents, non_parents, p_mutate, population_size):
    """Reproduce new individuals from parents"""

    new_population = []
    for idx in range(0, len(parents), 2):
        parent_i_genotype = parents[idx]["genotype"]
        parent_j_genotype = parents[idx+1]["genotype"]

        # crossover
        genotype_i, genotype_j = Crossover.one_point(
            cast("list[float]", parent_i_genotype),
            cast("list[float]", parent_j_genotype)
        )

        child_i = {"genotype": mutate(np.array(genotype_i), p_mutate), "fitness": None}
        child_j = {"genotype": mutate(np.array(genotype_j), p_mutate), "fitness": None}
        new_population.extend([parents[idx], parents[idx+1], child_i, child_j])
        i = 0 
        while len(new_population) < population_size:
            new_population.append(non_parents[i])
            i+=1

    return new_population


def mutate(genotype, p_mutate):
    connections = genotype[::2]
    mut_connections = random_swap(
        individual=cast("list[int]", connections),
        low=0,
        high=1,
        mutation_probability=p_mutate
    )
    genotype[::2] = np.array(mut_connections, dtype='float')
    weights = genotype[1::2]
    mut_weights = IntegerMutator.float_creep(
        individual=cast("list[float]", weights),
        span=1,
        mutation_probability=p_mutate
    )
    genotype[1::2] = np.array(mut_weights)
    return genotype

def fitness_function(movement_history) -> float:
    """The further from the center the better"""

    return np.linalg.norm(movement_history[-1] - movement_history[0])

def run_individual_trial(genome, hio, duration = 10.0) -> float:
    """Run a single trial of the genome"""
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()     
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    # No need to change anything above this line (i think)
    
    HISTORY.clear()

    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track, genome, hio))

    # Running the simulation without viewer
    simple_runner(
        model=model,
        data=data,
        duration=duration,
    )

    # print("Fitness:", fitness_function(HISTORY))

    return fitness_function(HISTORY)

def sigmoid(x):
    return (1/1+np.exp(-x))-0.5


def generate_genome(hio):
    hidden, input_nodes, output_nodes= hio
    genome = np.array([[np.random.randint(2), np.random.randn()] for _ in range(hidden* (input_nodes + output_nodes))]).flatten()
    return genome


def get_weights(genome, hio):
    hidden, input_nodes, output_nodes = hio
    w1 = np.array([genome[i+1] if genome[i] else 0 for i in range(0, genome.size // 2, 2)]).reshape(input_nodes, hidden)
    w2 = np.array([genome[i+1] if genome[i] else 0 for i in range(genome.size // 2, genome.size , 2)]).reshape(hidden, output_nodes)
    return w1, w2

#TODO
def controller(model, data, to_track, genome, hio) -> None:
    """Function to make the model move"""
    w1, w2 = get_weights(genome, hio)
    x = data.ctrl
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    data.ctrl += a2 * 0.01 
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
    HISTORY.append(to_track[0].xpos.copy())
    return None


def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints.
    
    The mujoco.set_mjcb_control() function will always give 
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu 
    
    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                   high=hinge_range, # pi/2
                                   size=num_joints) 

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta 

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    # 
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################

def show_qpos_history(history:list):
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


    HISTORY.clear() # I don't know why I re-use HISTORY, but whatever
    hio = (5,8,8)
    best_genome = train_model(
        number_of_generations = 1,
        population_size = 20,
        training_duration = 10.0, # Training duration refers to how long each individual is tested for in each trial. TODO: A better variable name?
        hio=hio
    )
    # Set the control callback function
    # This is called every time step to get the next action. 
    # Probably something like this to use our controller 
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track, best_genome, hio))

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
       model=model,  # type: ignore
       data=data,
    )

    # print("Fitness:", fitness_function(HISTORY))

    show_qpos_history(HISTORY)
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )

if __name__ == "__main__":
    main()