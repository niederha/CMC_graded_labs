""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np
import math

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
plt.rc('axes', titlesize=14.0)      # fontsize of the axes title
plt.rc('axes', labelsize=14.0)      # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)     # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)     # fontsize of the tick labels
plt.figure(figsize=(8.0, 5.0))      # figure size in inches


class TimeParameters:
    """Used to pass the time infos for the simulation in an easier way"""

    def __init__(self, t_start=0.0, t_stop=0.2, t_step=0.001):
        self.t_start = t_start
        self.t_stop = t_stop
        self.step = t_step
        self.times = np.arange(self.t_start, self.t_stop, self.step)


def find_ce_stretch_iso(muscle_sys, ce_stretch, time_param, stimulation=1, error_max=0.01):
    """Finds the total stretch [m] to apply to obtain a certain ce_stretch(relative) in isometric mode"""

    stretch = 0
    # The stretch is constant for over compression (i.e[0,sth]) hence we wanna monitor this to avoid being
    # in a regime where the computations are biased.
    minimal_ce_stretch = None
    x0 = [0.0, muscle_sys.muscle.L_OPT]

    while True:
        result = muscle_sys.integrate(x0=x0,
                                      time=time_param.times,
                                      time_step=time_param.step,
                                      stimulation=stimulation,
                                      muscle_length=stretch)
        if minimal_ce_stretch is None:
            minimal_ce_stretch = result.l_ce[-1]  # Monitoring the what the ce_stretch is in over compression
        else:
            if result.l_ce[-1] > ce_stretch * x0[1] and result.l_ce[-1] > minimal_ce_stretch:
                pylog.info("Reaches l_ce stretch " + str(result.l_ce[-1]/muscle_sys.muscle.L_OPT) +
                           " for total stretch of:" + str(stretch))
                break
            else:
                stretch += error_max

    return stretch


def plot_all_force_vs_stretch(active_force, passive_force, total_force, stretch, title, x_label, handle=None,
                              stimulation=1.):
    """
    Plots the 3 forces (active, passive and total) against the stretch.
    Parameters:
        - title: title of the graph
        - x_label: x_label of the graph
        - handle: hadle of the graph. If None title will be used as handle
    """
    force_labels = ['Active force, stimulation = {}'.format(stimulation),
                    'Passive, stimulation = {}'.format(stimulation),
                    'Total, stimulation = {}'.format(stimulation)]
    if handle is None:
        handle = title

    plt.figure(handle)
    plt.plot(stretch, active_force)
    plt.plot(stretch, passive_force)
    plt.plot(stretch, total_force)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Force [N]')
    plt.legend(force_labels)
    plt.grid()


def add_force_to_plot(handle, force, l):
    plt.figure(handle)
    plt.plot(l, force)
    plt.grid()


def plot_isometric_data(active_force, passive_force, total_force, l_ce, l_slack, l_mtc, handle, l_opt=None):
    """
    Plots three graphs (function of l_ce, l_mtc and l_ce)
    """
    plot_all_force_vs_stretch(active_force, passive_force, total_force, l_ce,
                              'Isometric muscle experiment: length of contractile element vs force',
                              'Stretch of the contractile element [m]',
                              handle=handle[0])
    plot_all_force_vs_stretch(active_force, passive_force, total_force, l_slack,
                              'Isometric muscle experiment: length of SLACK vs force', 'SLACK stretch [m]',
                              handle=handle[1])
    plot_all_force_vs_stretch(active_force, passive_force, total_force, l_mtc,
                              'Isometric muscle experiment: stretch of TOTAL MUSCLE vs force', 'TOTAL stretch [m]',
                              handle=handle[2])
    add_l_opt_marker(handle[0], l_opt)


def add_l_opt_marker(handle, l_opt=None):

    if l_opt is None:
        muscle_parameters = MuscleParameters()
        l_opt = muscle_parameters.l_opt
    plt.figure(handle)
    plt.axvline(x=l_opt, color="k")
    plt.text(l_opt+0.001, 2000, "L_opt", rotation=-90, color="k")


def get_common_axis_limits(handles):
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    for handle in handles:
        plt.figure(handle)
        x_bot, x_top = plt.xlim()
        y_bot, y_top = plt.ylim()
        x_min = min(x_min, x_bot)
        x_max = max(x_max, x_top)
        y_min = min(y_min, y_bot)
        y_max = max(y_max, y_top)
    return x_min, x_max, y_min, y_max


def iso_experiment(muscle_stimulation=1., ce_stretch_max=1.5, ce_stretch_min=0.5, nb_pts=1000,
                   time_param=TimeParameters(), l_opt=None):

    # System definition
    parameters = MuscleParameters()
    if l_opt is not None:
        parameters.l_opt = l_opt
    pylog.info(parameters.showParameters())
    muscle = Muscle(parameters)
    sys = IsometricMuscleSystem()
    sys.add_muscle(muscle)

    # Experiment parameters
    muscle_stretch_max = find_ce_stretch_iso(sys, ce_stretch_max, time_param)
    muscle_stretch_min = find_ce_stretch_iso(sys, ce_stretch_min, time_param)
    stretches = np.arange(muscle_stretch_min, muscle_stretch_max, muscle_stretch_max / nb_pts)
    x0 = [0.0, sys.muscle.L_OPT]

    # Containers
    active_force = []
    passive_force = []
    total_force = []
    l_ce = []
    l_slack = []
    l_mtc = []

    # Experiences
    for stretch in stretches:
        result = sys.integrate(x0=x0,
                               time=time_param.times,
                               time_step=time_param.step,
                               stimulation=muscle_stimulation,
                               muscle_length=stretch)
        active_force.append(result.active_force[-1])
        passive_force.append(result.passive_force[-1])
        total_force.append(result.tendon_force[-1])
        l_ce.append(result.l_ce[-1])
        l_mtc.append(result.l_mtc[-1])
        l_slack.append(result.l_mtc[-1]-result.l_ce[-1])

    return active_force, passive_force, total_force, l_ce, l_slack, l_mtc


def exercise1a(time_param):
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # region part_a
    pylog.info("Part a")
    # Parameters
    muscle_stimulation = 1.
    ce_stretch_max = 1.5
    ce_stretch_min = 0.5
    nb_pts = 1000
    handles = ["1_a_lce_vs_str", "1_a_lsl_vs_str", "1_a_lmtc_vs_str"]
    # Experiment
    active_force, passive_force, total_force, l_ce, l_slack, l_mtc = iso_experiment(muscle_stimulation, ce_stretch_max,
                                                                                    ce_stretch_min, nb_pts, time_param,)
    # Plotting
    plot_isometric_data(active_force, passive_force, total_force, l_ce, l_slack, l_mtc, handle=handles)
    # endregion


def exercise1b(time_param):
    pylog.info("part b")

    # region Parameters
    ce_stretch_max = 1.5
    ce_stretch_min = 0.5
    nb_pts = 1000
    muscle_stimulation_min = 0
    muscle_stimulation_max = 1
    step_stimulation = 0.25
    stimulations = np.arange(muscle_stimulation_min, muscle_stimulation_max, step_stimulation)
    # endregion

    # region Figure initialisation
    figure_handles = ["1_b_lce_stimulation", "1_b_lsl_stimulation", "1_b_lmtc_stimulation"]
    figure_titles = ["Isometric experiment: length of contractile element vs force",
                     "Stimulation variation SLACK stretch vs force",
                     "Stimulation variation total stretch vs force"]
    figure_x_label = ["Stretch of the contractile element [m]", "SLACK stretch [m]", "TOTAL stretch [m]"]
    force_legends = []
    for handle, title, x_label in zip(figure_handles, figure_titles, figure_x_label):
        plt.figure(handle)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('Force [N]')
        plt.grid()
    # endregion

    # region Experiences
    for stim in stimulations:
        active_force, passive_force, total_force, l_ce, l_slack, l_mtc = iso_experiment(stim,
                                                                                        ce_stretch_max,
                                                                                        ce_stretch_min, nb_pts,
                                                                                        time_param)
        # adding active force
        lengths = [l_ce, l_slack, l_mtc]
        force_legends.append("Active force, stimulation={}".format(stim))
        # Plots
        for handle, length in zip(figure_handles, lengths):
            add_force_to_plot(handle, active_force, length)

    # Adding passive force only once
    for handle, length in zip(figure_handles, lengths):
        add_force_to_plot(handle, passive_force, length)
    force_legends.append("Passive force (not stimulation dependent)")

    for handle in figure_handles:
        plt.figure(handle)
        plt.legend(force_legends)
        plt.grid()

    add_l_opt_marker(figure_handles[0])
    # endregion


def exercise1c(time_param):
    pylog.info("part C")
    l_opt_long = 0.5
    l_opt_small = 0.25
    stim = 1.
    ce_stretch_max = 1.5
    ce_stretch_min = 0.5
    nb_pts = 1000
    handles_long = ["1_c_lce_iso_long", "1_c_lsl_iso_long", "1_c_lmtc_long"]
    handles_small = ["1_c_lce_iso_small", "1_c_lsl_iso_small", "1_c_lmtc_small"]
    handles_merged = ["1_c_lce_iso_merged", "1_c_lsl_iso_merged", "1_c_lmtc_merged"]
    merged_legends = ["Active force, l_opt=0.5", "Passive force, l_opt=0.5", "Total force, l_opt=0.5", "_nolegend_",
                      "Active force, l_opt=0.25", "Passive force, l_opt=0.25", "Total force, l_opt=0.25"]
    active_force_long, passive_force_long, total_force_long, l_ce_long, l_slack_long, l_mtc_long = \
        iso_experiment(stim, ce_stretch_max, ce_stretch_min, nb_pts, time_param, l_opt_long)
    active_force_small, passive_force_small, total_force_small, l_ce_small, l_slack_small, l_mtc_small = \
        iso_experiment(stim, ce_stretch_max, ce_stretch_min, nb_pts, time_param, l_opt_small)

    # Separated plots
    plot_isometric_data(active_force_long, passive_force_long, total_force_long, l_ce_long, l_slack_long, l_mtc_long,
                        handle=handles_long, l_opt=l_opt_long)
    plot_isometric_data(active_force_small, passive_force_small, total_force_small, l_ce_small, l_slack_small,
                        l_mtc_small, handle=handles_small, l_opt=l_opt_small)

    # Merged Plots
    plot_isometric_data(active_force_long, passive_force_long, total_force_long, l_ce_long, l_slack_long, l_mtc_long,
                        handle=handles_merged, l_opt=l_opt_long)
    plot_isometric_data(active_force_small, passive_force_small, total_force_small, l_ce_small, l_slack_small,
                        l_mtc_small, handle=handles_merged, l_opt=l_opt_small)

    for handle_long, handle_small, handle_merged in zip(handles_long, handles_small, handles_merged):
        x_min, x_max, y_min, y_max = get_common_axis_limits([handle_small, handle_long])
        plt.figure(handle_merged)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid()
        plt.legend(merged_legends, prop={'size': 8})


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
    pylog.info("Start exercise 1")
    time_param = TimeParameters(0.0, 0.2, 0.001)

    # exercise1a(time_param)
    # exercise1b(time_param)
    exercise1c(time_param)
    # exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
        plt.show()


if __name__ == '__main__':
    parse_args()
    exercise1()

