""" Lab 6 Exercise 3
This file implements the pendulum system with two muscles attached driven
by a neural network
"""

import numpy as np
from matplotlib import pyplot as plt

import cmc_pylog as pylog
from time_parameters import TimeParameters
from cmcpack import DEFAULT
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
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def plot_results(res, prefix):
    """ PLots every important graph for a simulation an labels them using the prefix"""

    activation_handle = prefix+'_activations'
    state_handle = prefix+'_state'
    phase_handle = prefix+'_phase'

    # Activations
    plt.figure(activation_handle)
    plt.title('Oscillating activations through time')
    plt.plot(res[:, 0], res[:, 3], label="Activation 1")
    plt.plot(res[:, 0], res[:, 5], label="Activation 2")
    plt.xlabel('Time [s]')
    plt.ylabel('Activation intensity')
    plt.legend()
    plt.grid()

    # Plotting the results
    plt.figure(state_handle)
    plt.title('State of the pendulum')
    plt.plot(res[:, 0], res[:, 1])
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\theta$[rad]')
    plt.grid()

    # Plotting the results
    plt.figure(phase_handle)
    plt.title('Phase of the pendulum')
    plt.plot(res[:, 1], res[:, 2])
    plt.xlabel(r'$\theta$[rad]')
    plt.ylabel(r'$\dot{\theta}$[rad]')
    plt.grid()
    return activation_handle, state_handle, phase_handle

def system_initialisation(l_pendulum=0.5, m_pendulum=1., f_max=1500, l_attach=0.17):
    """Generates a oscillatory system and its default initial conditions"""

    # Neural parameters for oscillatory system
    d = 1.
    w = np.array([[0, -5, -5, 0],
                           [-5, 0, 0, -5],
                           [5, -5, 0, 0],
                           [-5, 5, 0, 0]])
    b = np.array([3., 3., -3., -3.])
    tau = np.array([0.02, 0.02, 0.1, 0.1])

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

    # Neural network
    n_params = NetworkParameters()
    n_params.D = d
    n_params.w = w
    n_params.b = b
    n_params.tau = tau
    neural_network = NeuralSystem(n_params)

    # System creation
    sys = System()                         # Instantiate a new system
    sys.add_pendulum_system(pendulum)      # Add the pendulum model to the system
    sys.add_muscle_system(muscles)         # Add the muscle model to the system
    sys.add_neural_system(neural_network)  # Add neural network model to the system

    # Default initial conditions
    x0_p = np.array([0, 0.])                # Pendulum initial condition
    x0_m = np.array([0., m1.L_OPT, 0., m2.L_OPT])   # Muscle Model initial condition
    x0_n = np.array([-0.5, 1, 0.5, 1])              # Neural Network initial condition
    x0 = np.concatenate((x0_p, x0_m, x0_n))         # System initial conditions
    return sys, x0


def exercise3a(time_param):
    """ Instantiate simulates and plots a system in driven by an oscillatory neural network"""

    # Simulation
    sys, x0 = system_initialisation()
    sim = SystemSimulation(sys)
    sim.initalize_system(x0, time_param.times)
    sim.simulate()
    res = sim.results()  # [time, states]

    # Plots
    plot_results(res, '3_a')


def exercise3b(time_param):
    """ Simulate the system multiple times by applying various external drives and plots the results"""
    drive_min = 0
    drive_max = 1
    nb_drives = 3
    drives = np.linspace(drive_min, drive_max, nb_drives)

    for drive in drives:
        # Simulate for drive
        sys, x0 = system_initialisation()
        sim = SystemSimulation(sys)
        sim.initalize_system(x0, time_param.times)
        ext_drive = np.ones((time_param.nb_pts, 4))*drive
        ext_drive[0:int(time_param.nb_pts/4), :] *= 0
        sim.add_external_inputs_to_network(ext_drive)
        sim.simulate()
        res = sim.results()

        # Plot
        activation_handle, state_handle, phase_handle = plot_results(res, '3_b_'+str(drive))

        # Add marker where external drive was applied
        plt.figure(activation_handle)
        plt.axvline(time_param.times[int(time_param.nb_pts/4)], color='k')
        plt.text(time_param.times[int(time_param.nb_pts/4)]+0.1, 0.25, "External input", rotation=-90, color="k")
        plt.figure(state_handle)
        plt.axvline(time_param.times[int(time_param.nb_pts / 4)], color='k')
        plt.text(time_param.times[int(time_param.nb_pts / 4)] + 0.1, -0.02, "External input", rotation=-90, color="k")


def exercise3():

    time_param = TimeParameters(time_start=0, time_stop=10., time_step=0.001)
    exercise3a(time_param)
    exercise3b(time_param)

    if DEFAULT["save_figures"]:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
    plt.show()


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise3()

