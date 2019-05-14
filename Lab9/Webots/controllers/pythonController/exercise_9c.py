"""Exercise 9c"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9c(world, timestep, reset):
    """Exercise 9c"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=4,
            amplitudes=0.5,
            frequency=1,
            RHead = RHead,
            RTail = RTail,
            Backwards = False,
            phase_lag=2*np.pi/10,
            turnRate=[1,1],
            # ...
        )
        for RHead in np.linspace(0,1, 5)
        for RTail in np.linspace(0,1, 5)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/exercise_9c/simulation_{}.npz".format(simulation_i)
        )

