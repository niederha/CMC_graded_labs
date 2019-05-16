"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    # Parameters
    mlr_drive = 4.  # Experiment for swimming only

    parameter_set = [
        SimulationParameters(
            simulation_duration=30,
            mlr_drive=mlr_drive,
        )
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9b_walking/simulation_{}.npz".format(simulation_i)
        )



