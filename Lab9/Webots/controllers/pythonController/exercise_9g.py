"""Exercise 9g"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters



def exercise_9g(world, timestep, reset):
    """Exercise 9g"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=40,
            drive=2,
        )
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/test/simulation_{}.npz".format(simulation_i)
        )
