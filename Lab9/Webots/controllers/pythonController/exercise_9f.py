"""Exercise 9f"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    n_joints = 10
    
    #ex9f1
    
#    parameter_set = [
#        SimulationParameters(
#            simulation_duration=10,
#            drive=1,
#            amplitudesLimb=0.3,
#            phase_lag_body_limb=phase_lag_body_limb,
#        )
#        for phase_lag_body_limb in np.linspace(0,2*np.pi,30)
#    ]
    
    #ex9f2
#    parameter_set = [
#        SimulationParameters(
#            simulation_duration=10,
#            drive=1,
#
#            amplitudesLimb=amplitude,
#            phase_lag_body_limb = 2.816
#        )
#        for amplitude in np.linspace(0,0.5,30)
#    ]
    
    #normal walking
    parameter_set = [
        SimulationParameters(
            simulation_duration=40,
            drive=1,

            #amplitudesLimb=0.32,
            #phase_lag_body_limb = 2.816,
        )
        #for amplitude in np.linspace(0,0.5,15)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/exercise_9f_walking/simulation_{}.npz".format(simulation_i)
        )

