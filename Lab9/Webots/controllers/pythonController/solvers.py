"""ODE solvers using fixed step integration"""
import cmc_pylog as pylog
import numpy as np

def euler(ode, timestep, time, state, *parameters):
    """ODE solver with euler method"""
    return timestep*ode(time, state, *parameters)


def rk4(ode, timestep, time, state, *parameters):
    """ODE solver with Runge-Kutta method"""
    k_1 = timestep*ode(time, state, *parameters)
    pylog.warning(np.shape(k_1))
    k_2 = timestep*ode(time+0.5*timestep, state+0.5*k_1, *parameters)
    k_3 = timestep*ode(time+0.5*timestep, state+0.5*k_2, *parameters)
    k_4 = timestep*ode(time+timestep, state+k_3, *parameters)
    return (k_1+2*k_2+2*k_3+k_4)/6

