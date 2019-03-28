""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def find_ce_stretch_iso(ce_stretch, t_start, t_stop, dt, stimulation=1, error_max=0.01):
    """Finds the total relative stretch to apply to obtain ce_stretch in isometric mode"""
    stretch = error_max

    # Muscle definition
    parameters = MuscleParameters()
    muscle = Muscle(parameters)
    sys = IsometricMuscleSystem()
    sys.add_muscle(muscle)

    # Simulation parameters
    x0 = [0.0, sys.muscle.L_OPT]
    time = np.arange(t_start, t_stop, dt)

    while True:
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=dt,
                               stimulation=stimulation,
                               muscle_length=stretch * (sys.muscle.L_OPT+sys.muscle.L_SLACK))
        if result.l_ce[-1]/sys.muscle.L_OPT > ce_stretch:
            pylog.info("Reaches l_ce stretch " + str(result.l_ce[-1]/sys.muscle.L_OPT) + " for total stretch of:" +
                       str(stretch))
            break
        else:
            stretch += error_max

    return stretch


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Simulation parameters
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)

    # Definition of system
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    muscle = Muscle(parameters)
    sys = IsometricMuscleSystem()
    sys.add_muscle(muscle)

    # Experiences parameters
    muscle_stimulation = 1.
    ce_stretch_max = 2.
    nb_pts = 1000
    muscle_stretch_max = find_ce_stretch_iso(ce_stretch_max, t_start, t_stop, time_step)
    stretches = np.arange(0, muscle_stretch_max, muscle_stretch_max/nb_pts)
    x0 = [0.0, sys.muscle.L_OPT+sys.muscle.L_SLACK]

    active_force = []
    passive_force = []
    total_force = []
    l_ce = []
    l_t = []
    l_mtc = []

    for stretch in stretches:
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=muscle_stimulation,
                               muscle_length=stretch * (x0[1]))
        active_force.append(result.active_force[-1])
        passive_force.append(result.passive_force[-1])
        total_force.append(result.tendon_force[-1])
        l_ce.append(result.l_ce[-1])
        l_mtc.append(result.l_mtc[-1])
        l_t.append(l_mtc[-1]-l_ce[-1])

    # Plotting
    plt.figure('Isometric muscle experiment')
    plt.plot(result.time, result.tendon_force)
    plt.title('Isometric muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle Force')
    plt.grid()


def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Definition of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instantiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evaluate for a single load
    load = 100.

    # Evaluate for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           time_stabilize=time_stabilize,
                           stimulation=muscle_stimulation,
                           load=load)

    # Plotting
    plt.figure('Isotonic muscle experiment')
    plt.plot(result.time, result.v_ce)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractilve velocity')
    plt.grid()


def exercise1():
    plt.close("all")
    exercise1a()
    # exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

