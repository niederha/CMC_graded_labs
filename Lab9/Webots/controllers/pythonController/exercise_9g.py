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
            drive=drive,
            turn=0,
            # ...
        )
        for drive in np.linspace(1, 5, 2)
        # for amplitudes in ...
        # for ...
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/exercise_9g/simulation_{}.npz".format(simulation_i)
        )
