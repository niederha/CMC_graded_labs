""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt, cos, sin, pi

import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
from time_parameters import TimeParameters

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels
plt.rcParams['figure.figsize'] = [9, 6]  # figure size in inches


def compute_muscle_length(a1, a2, theta):
    return sqrt(a1**2+a2**2+2*a1*a2*sin(theta))


def compute_moment_arm(a1, a2, theta, l1):
    return a1*a2*cos(theta)/l1


def exercise2a():
    """Computes and plots the muscle length and moment arm as a theoretical function of theta"""
    pylog.info("Exercise 2a")

    # Parameters
    a1 = 1
    ratio_min = 0.25
    ratio_max = 2
    nb_ratios = 8
    ratios = np.linspace(ratio_min, ratio_max, nb_ratios)
    theta_min = -pi/4
    theta_max = pi/4
    nb_thetas = 1000
    thetas = np.linspace(theta_min, theta_max, nb_thetas)
    handle_mscl_length = "2_a_muscle_length"
    handle_mmt_arm = "2_a_moment_arm"

    # Data containers
    muscle_length = []
    moment_arm = []
    muscle_length_3d = np.zeros((nb_ratios, nb_thetas))
    moment_arm_3d = np.zeros((nb_ratios, nb_thetas))
    ratios_3d = np.ones((nb_ratios, nb_thetas))

    # Graph initialisations
    plt.figure(handle_mscl_length)
    plt.title("Muscle length vs theta")
    plt.xlabel("Theta [rad]")
    plt.ylabel("Muscle length [m]")
    plt.figure(handle_mmt_arm)
    plt.title("Moment arm vs theta")
    plt.xlabel("Theta [rad]")
    plt.ylabel("Moment arm [m]")

    # Computations
    for i, ratio in enumerate(ratios):
        a2 = a1*ratio
        for theta in thetas:
            muscle_length.append(compute_muscle_length(a1, a2, theta))
            moment_arm.append(compute_moment_arm(a1, a2, theta, muscle_length[-1]))

        # Fill up 3d containers
        muscle_length_3d[i, :] = muscle_length
        moment_arm_3d[i, :] = moment_arm
        ratios_3d[i, :] *= ratio

        # 2D plots
        plt.figure(handle_mscl_length)
        plt.plot(thetas, muscle_length, label="a2/a1={}".format(ratio))
        plt.figure(handle_mmt_arm)
        plt.plot(thetas, moment_arm, label="a2/a1={}".format(ratio))

        # Clear
        muscle_length = []
        moment_arm = []

    # finish 2d plots
    plt.figure(handle_mscl_length)
    plt.legend()
    plt.grid()
    plt.figure(handle_mmt_arm)
    plt.legend()
    plt.grid()

    # 3d plot
    fig = plt.figure('2_a_3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Ratio vs muscle length vs moment arm")
    ax.plot_surface(ratios_3d, muscle_length_3d, moment_arm_3d, cmap=cm.coolwarm)
    ax.set_xlabel('Ratio a2/a1 [-]')
    ax.set_ylabel('Muscle length [m]')
    ax.set_zlabel('Moment arm [m]')


def system_initialisation(l_pendulum=0.5, m_pendulum=1., f_max=1500, l_attach=0.17):
    """Generates a system and its default initial conditions"""

    # Pendulum parameters
    pendulum_params = PendulumParameters()
    pendulum_params.L = l_pendulum
    pendulum_params.m = m_pendulum
    pendulum = PendulumSystem(pendulum_params)

    # Muscles parameters
    m1_param = MuscleParameters()
    m1_param.f_max = f_max
    m2_param = MuscleParameters()
    m2_param.f_max = f_max
    m1 = Muscle(m1_param)
    m2 = Muscle(m2_param)
    muscles = MuscleSytem(m1, m2)

    # Muscle_attachment
    m1_origin = np.array([-l_attach, 0.0])
    m1_insertion = np.array([0.0, -l_attach])
    m2_origin = np.array([l_attach, 0.0])
    m2_insertion = np.array([0.0, -l_attach])
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    # System creation
    sys = System()                      # Instantiate a new system
    sys.add_pendulum_system(pendulum)   # Add the pendulum model to the system
    sys.add_muscle_system(muscles)      # Add the muscle model to the system

    # Default initial conditions
    x0_p = np.array([np.pi / 4, 0.])                # Pendulum initial condition
    x0_m = np.array([0., m1.L_OPT, 0., m2.L_OPT])   # Muscle Model initial condition
    x0 = np.concatenate((x0_p, x0_m))               # System initial conditions
    return sys, x0


def generate_sinus_activation(frq, times):
    """ Generates a sine wave activation over times[s](array-like) with frequency frq[Hz]"""
    act1 = np.ones((len(times), 1))

    for i, time in enumerate(times):
        act1[i] = sin(2*pi*frq)
    act2 = act1*-1

    for i in range(len(times)):
        if act1[i] < 0:
            act1[i] = 0
        if act2[i] < 0:
            act2[i] = 0

    return act1, act2

def plot_quiver(x,y, nb_meshes, act1, act2):

    # Quiver mesh
    x_pos = np.linspace(x.min(), x.max(), nb_meshes)
    y_pos = np.linspace(y.min(), y.max(), nb_meshes)
    u_pos, v_pos = np.meshgrid(x_pos, y_pos)
    x_intensity = []
    y x_intensity = []
    # Short execution time to evalute the quiver
    time_param_short = time_param(t_stop=0.003, t_step=0.001)
    act = np.hstack((act1[0:time_param.nb_pts-1], act2[0:time_param.nb_pts-1]))

    sys,x0 = system_initialisation()



def exercise2b(time_param):
    pylog.info("Exercise 2b")

    # Parameters
    activation_frq = 5

    # Initialisation
    sys, x0 = system_initialisation()
    sim = SystemSimulation(sys)

    # Activation
    act1, act2 = generate_sinus_activation(activation_frq, time_param.times)
    activations = np.hstack((act1, act2))
    sim.add_muscle_activations(activations)

    # Simulation
    sim.initalize_system(x0, time_param.times)
    sim.sys.pendulum_sys.parameters.PERTURBATION = True
    sim.simulate()
    res = sim.results()  # [time, states]

    plt.figure("2_b")

def exercise2():

    time_param = TimeParameters(time_start=0, time_stop=2.5, time_step=0.001)

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
    x0_p = np.array([np.pi/4, 0.])                 # Pendulum initial condition
    x0_m = np.array([0., M1.L_OPT, 0., M2.L_OPT])  # Muscle Model initial condition
    x0 = np.concatenate((x0_p, x0_m))              # System initial conditions
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
    res = sim.results()  # [time, states]
    muscle1_results = sim.sys.muscle_sys.Muscle1.results
    muscle2_results = sim.sys.muscle_sys.Muscle2.results
    # endregion
    # endregion

    # region plot
    plt.figure('Pendulum')
    plt.title('Pendulum Phase')
    plt.plot(res[:, 1], muscle1_results.l_mtc)
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    simulation = SystemAnimation(res, pendulum, muscles)
    if DEFAULT["save_figures"] is False:
        simulation.animate()
    # endregion


if __name__ == '__main__':
    plt.close("all")
    parse_args()
    time_param = TimeParameters(time_start=0, time_stop=5, time_step=0.001)
    # exercise2a()
    exercise2b(time_param)
    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
        plt.show()

