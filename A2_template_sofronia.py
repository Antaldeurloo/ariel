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
FITNESS = []


# Global flag to track which simulation is running
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
    max_range = max(abs(pos_data).max(), 0.3) 
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    plt.show()


def fitness_function(movment_history) -> float:
    """The further from the center the better"""
    return -1 * (movment_history[-1][1] - movment_history[1][0]) - abs(movment_history[-1][0] - movment_history[0][0])**2


# CPG-based controller
def controller(model, data, cpg_params) -> None:
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
    t = data.time
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
    cpg_params = np.array(individual)
    mujoco.set_mjcb_control(lambda m, d: controller(m, d, cpg_params))

    simple_runner(
        model=model,
        data=data,
        duration=duration,
    )

    fitness = fitness_function(HISTORY)

    return (fitness,)



def train_model(toolbox, stats, number_of_generations=10, population_size=5,
                training_duration=10.0, cxpb=0.5, mutpb=0.2):
    """Training model using the DEAP library"""

    # Generate the initial population
    population = toolbox.population(n=population_size)

    # The DEAP's evolutionary loop.
    pop, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb, # Probability of crossover
        mutpb=mutpb, # Probability of mutation
        ngen=number_of_generations,
        stats=stats,
        verbose=False
    )
    print(logbook)

    # Return the best individual from the final population
    return tools.selBest(population, 1)[0], logbook

def run_evolution(mutpb, cxpb, iters, training_duration, population_size, num_generations, export_video=True):
    plt.figure(figsize=(6,5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    track = []
    for i in range(iters):
        """Main function to run the simulation with random movements."""
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

        # Data tracking
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('max', np.max)

        # Train model
        best_individual, logbook = train_model(
            toolbox,
            stats,
            training_duration=training_duration,
            population_size=population_size,
            number_of_generations=num_generations,
            mutpb=mutpb,
            cxpb=cxpb
        )
        avg, std, max_ = logbook.select('avg', 'std', 'max')
        avg = np.array(avg)
        std = np.array(std)
        track.append([avg, std, max_])
        x = np.arange(len(avg))
        plt.plot(x, avg, color=colors[i], linestyle='dashed', label='mean run: {i}')
        plt.fill_between(x, avg+std, avg-std, color=colors[i], alpha=0.5, label=f'mean±std run: {i}')
        plt.plot(x, max_, color=colors[i], label='max run: {i}')

        if export_video:
            mujoco.set_mjcb_control(None)
            world = SimpleFlatWorld()
            gecko_core = gecko()
            world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
            model = world.spec.compile()
            data = mujoco.MjData(model)
            cpg_params = np.array(best_individual)
            mujoco.set_mjcb_control(lambda m, b: controller(m, b, cpg_params))
            PATH_TO_VIDEO_FOLDER = "./__videos__"
            video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=10,
                video_recorder=video_recorder,
            )

    plt.legend()
    plt.savefig(f'chart_{mutpb}_{cxpb}.png')

    track = np.array(track)
    avg = track[:, 0]
    std = track[:, 1]
    max_overall = track[:, 2]
    print(avg.shape, std.shape, max_overall.shape)
    d0 = avg[0] - (avg[0] + avg[1] + avg[2])/3
    d1 = avg[1] - (avg[0] + avg[1] + avg[2])/3
    d2 = avg[2] - (avg[0] + avg[1] + avg[2])/3
    sig_0 = std[0]
    sig_1 = std[1]
    sig_2 = std[2]
    sig_123 = np.sqrt(population_size * (np.square(sig_0) + np.square(sig_1) + np.square(sig_2) + np.square(d0) + np.square(d1) + np.square(d2)) / (3 * population_size))
    print(sig_123.shape)
    avg_123 = (avg[0] + avg[1] + avg[2]) / 3
    x = np.arange(len(avg_123))
    plt.figure(figsize=(6,5))
    plt.plot(x, avg_123, color=colors[i], linestyle='dashed', label='mean averaged')
    plt.fill_between(x, avg_123+sig_123, avg_123-sig_123, color=colors[i], alpha=0.5, label=f'mean±std')
    plt.plot(x, np.max(max_overall, axis=0), color=colors[i], label='max')
    plt.legend()
    plt.savefig(f'chart_{mutpb}_{cxpb}_averaged.png')



def main():
    np.random.seed(42)
    random.seed()
    mutation_probabilities = [0.0, 0.2, 0.7]
    for mutpb in mutation_probabilities:
        run_evolution(
            mutpb=mutpb,
            cxpb=0.5,
            iters=3,
            training_duration=10,
            population_size=20,
            num_generations=50,
            export_video=False
        )

if __name__ == "__main__":
    main()