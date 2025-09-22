# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# DEAP Toolbox
from deap import base, creator, tools, algorithms
import time

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

from ariel.utils.runners import simple_runner

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []

# Global flag to track which simulation is running
IS_RANDOM_RUN = False

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
    max_range = max(abs(pos_data).max(), 0.3) 
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()

def fitness_function(movment_history) -> float:
    """The further from the center the better"""
    final_position = movment_history[-1]
    return final_position[0]

# CPG-based controller
def controller(model, data, cpg_params, step_count) -> None:
    """
    Function to make the model move based on the provided CPG parameters.
    The genome is used to set the CPG parameters.
    """
    num_joints = model.nu
    hinge_range = np.pi/2
    
    # The first 8 parameters are amplitudes, the last 8 are phases
    amplitudes = cpg_params[:num_joints]
    phases = cpg_params[num_joints:]
    
    # Generate an oscillating signal for each joint using the CPG parameters
    t = step_count * model.opt.timestep
    cpg_output = amplitudes * np.sin(2 * np.pi * t + phases)
    
    # Apply the CPG output to the robot's control signals
    data.ctrl += cpg_output
    
    # Bound the control values to be within the hinge limits.
    data.ctrl = np.clip(data.ctrl, -hinge_range, hinge_range)
    
    # Save movement to history
    HISTORY.append(data.geom("robot-core").xpos.copy())



# Evaluation function
def run_individual_trial(individual, duration = 10.0) -> float:
    """Run a single trial of the genome"""
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    HISTORY.clear()
    
    # The individual's genome is used to define the CPG parameters
    if not IS_RANDOM_RUN:
        cpg_params = np.array(individual)
    
    step_count = 0
    def control_with_cpg(m, d):
        nonlocal step_count
        if IS_RANDOM_RUN:
            random_move(m, d, [data.geom("robot-core")])
        else:
            controller(m, d, cpg_params, step_count)
            step_count += 1
    
    mujoco.set_mjcb_control(control_with_cpg)

    simple_runner(
        model=model,
        data=data,
        duration=duration,
    )

    fitness = fitness_function(HISTORY)
    print("Fitness:", fitness)

    return (fitness,)

# Fitness strategy for maximization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# The genome now has 16 dimensions (8 amplitudes + 8 phases)
creator.create("Individual", list, fitness=creator.FitnessMax)

# Using DEAP's tools
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
# Individual generator
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=16)
# Population generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators
toolbox.register("evaluate", run_individual_trial)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)


def train_model(number_of_generations = 10, population_size = 5, training_duration = 10.0):
    """Training model using the DEAP library"""
    global IS_RANDOM_RUN
    IS_RANDOM_RUN = False

    # Generate the initial population
    population = toolbox.population(n=population_size)

    # The DEAP's evolutionary loop.
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5, # Probability of crossover
        mutpb=0.2, # Probability of mutation
        ngen=number_of_generations,
        verbose=True
    )

    # Return the best individual from the final population
    return tools.selBest(population, 1)[0]

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

    # Set the control callback function
    # This is called every time step to get the next action. 
    mujoco.set_mjcb_control(lambda m,d: random_move(m, d, to_track))

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,  # type: ignore
        data=data,
    )

    show_qpos_history(HISTORY)
    # If you want to record a video of your simulation, you can use the video renderer.

    # Non-default VideoRecorder options
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