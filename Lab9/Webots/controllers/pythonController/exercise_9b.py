"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=4,
            amplitudes=amplitude,
            frequency=1,
            phase_lag=phase_lag,
        )
        for amplitude in np.linspace(0.1,0.5, 10)
        for phase_lag in np.linspace(0, 3, 30)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/exercise_9b_300/simulation_{}.npz".format(simulation_i)
        )

