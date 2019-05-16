"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=2,
            amplitudes=0.27,
            frequency=1,
            RHead = 1,
            RTail = 1,
            Backwards = False,
            phase_lag=0.27,
            turnRate=[1,1],
            # ...
        )
        #for amplitude in np.linspace(0.1,0.5, 10)
        #for phase_lag in np.linspace(0, 3, 30)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            #logs="./logs/exercise_9b/simulation_{}.npz".format(simulation_i)
            logs="./logs/test/simulation_{}.npz".format(simulation_i)
        )

