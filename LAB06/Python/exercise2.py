""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt

import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt

from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation


# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


class timeParamters:

    def __init__(self, t_init=0, t_final=2.5, t_step=0.001):
        self.t_init = t_init
        self.t_final = t_final
        self.t_step = t_step
        self.times = np.arange(t_init, t_final, t_step)
        self.nb_pts = len(self.times)


def exercise2():
    """ Main function to run for Exercise 2.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    time_param = timeParamters()

    # region system initialisation
    # region pendulum_definition
    pendulum_params = PendulumParameters()
    pendulum_params.L = 0.5
    pendulum_params.m = 1.
    pendulum = PendulumSystem(pendulum_params)
    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))
    # endregion

    # region muscle_definition
    M1_param = MuscleParameters()
    M1_param.f_max = 1500
    M2_param = MuscleParameters()
    M2_param.f_max = 1500
    M1 = Muscle(M1_param)
    M2 = Muscle(M2_param)
    muscles = MuscleSytem(M1, M2)
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))
    # endregion

    # region muscle_attachment
    m1_origin = np.array([-0.17, 0.0])
    m1_insertion = np.array([0.0, -0.17])
    m2_origin = np.array([0.17, 0.0])
    m2_insertion = np.array([0.0, -0.17])

    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))
    # endregion

    # region system_creation
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system
    # endregion
    # endregion

    # region initial_conditions
    x0_p = np.array([np.pi/4, 0.])  # Pendulum initial condition
    x0_m = np.array([0., M1.L_OPT, 0., M2.L_OPT])  # Muscle Model initial condition
    x0 = np.concatenate((x0_p, x0_m))  # System initial conditions
    # endregion

    # region system_simulation
    sim = SystemSimulation(sys)
    # region activations
    act1 = np.ones((time_param.nb_pts, 1)) * 1.
    act2 = np.ones((time_param.nb_pts, 1)) * 0.05
    activations = np.hstack((act1, act2))
    sim.add_muscle_activations(activations)
    # endregion

    # region integration
    sim.initalize_system(x0, time_param.times)  # Initialize the system state
    sim.sys.pendulum_sys.parameters.PERTURBATION = True
    sim.simulate()
    res = sim.results()
    muscle1_results = sim.sys.muscle_sys.Muscle1.results
    muscle2_results = sim.sys.muscle_sys.Muscle2.results
    # endregion
    # endregion

    # region plot
    plt.figure('Pendulum')
    plt.title('Pendulum Phase')
    plt.plot(res[:, 1], res[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    simulation = SystemAnimation(res, pendulum, muscles)
    if DEFAULT["save_figures"] is False:
        simulation.animate()
    # endregion
    
    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
        plt.show()


if __name__ == '__main__':
    plt.close("all")
    parse_args()
    exercise2()

