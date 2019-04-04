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


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    parameters.l_opt = 0.05 # and 0.5
    #pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Create muscle object
    muscle = Muscle(parameters)

    #pylog.warning("Isometric muscle contraction to be completed")

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single muscle stretch
    #muscle_stretch = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4])
    muscle_stretch = np.linspace(0.2,1.3,100)
    muscle_stretch = muscle_stretch*(sys.muscle.L_OPT+sys.muscle.L_SLACK)
    # Evalute for a single muscle stimulation
    muscle_stimulation = np.linspace(1.0,1.0,1)

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001

    time = np.arange(t_start, t_stop, time_step)


    # Run the integration
    for s in range(0,len(muscle_stimulation)): 
        l_ce = []
        active_force = []
        passive_force = []
        for i in range(0,len(muscle_stretch)):
            result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=muscle_stimulation[s],
                               muscle_length=muscle_stretch[i])
            l_ce.append(result.l_ce[-1])
            active_force.append(result.active_force[-1])
            passive_force.append(result.passive_force[-1])

    # Plotting
#    plt.figure('Isometric muscle experiment, Force vs Time')
#    plt.plot(result.time, result.active_force)
#    plt.plot(result.time, result.passive_force)
#    plt.title('Isometric muscle experiment, Force vs Time')
#    plt.xlabel('Time [s]')
#    plt.ylabel('Muscle Force')
#    plt.grid()
    
        plt.figure('Isometric muscle experiment, Force vs Length')
        plt.plot(l_ce, active_force, label='act F, stim = %0.1f'%muscle_stimulation[s])
        plt.plot(l_ce, passive_force, label='pass F, stim = %0.1f' %muscle_stimulation[s])
        plt.title('Isometric muscle experiment, Force vs Length')
        plt.xlabel('Contractile Element Length [m]')
        plt.ylabel('Muscle Force [N]')
        plt.grid()
        plt.legend()


def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed to contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single load
    load = np.linspace(20.0,300.0,200)

    # Evalute for a single muscle stimulation
    muscle_stimulation = np.linspace(0.0,1.0,4)

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
    
    for s in range(0,len(muscle_stimulation)): 
        v_ce = []
        tendon_force = []
        # Run the integration
        for i in range(0,len(load)):
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   time_stabilize=time_stabilize,
                                   stimulation=muscle_stimulation[s],
                                   load=load[i])
            tendon_force.append(result.tendon_force[-1])
            if (result.l_mtc[-1] > (sys.muscle.L_OPT + sys.muscle.L_SLACK)):
                v_ce.append(max(result.v_ce))
            else:
                v_ce.append(min(result.v_ce))
    
        # Plotting
        plt.figure('Isotonic muscle experiment, Contractile Velocity vs. Tendon Force')
        plt.plot(tendon_force, v_ce, label = 'stim = %0.1f'%muscle_stimulation[s])
        plt.title('Isotonic muscle experiment, Contractile Velocity vs. Tendon Force')
        plt.xlabel('Tendon force [N]')
        plt.ylabel('Muscle contractile velocity [m/s]')
        plt.grid()
        plt.legend()
    
    plt.figure('Isotonic muscle experiment, Contractile Velocity vs. Time')
    plt.plot(result.time, result.v_ce)
    plt.title('Isotonic muscle experiment, Contractile Velocity vs. Time')
    plt.xlabel('Tendon force [N]')
    plt.ylabel('Muscle contractile velocity [m/s]')
    plt.grid()

def exercise1():
    plt.close('all')
    plt.rcParams['figure.figsize'] = [9, 6]
    #exercise1a()
    exercise1d()

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
