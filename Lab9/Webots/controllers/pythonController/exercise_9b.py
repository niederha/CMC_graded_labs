"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    # Parameters
    n_joints = 10

    parameter_set = [
        SimulationParameters(
            simulation_duration=30,
            mlr_drive=5.,
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



