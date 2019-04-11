""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tqdm import tqdm
from enum import Enum, unique

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem
from time_parameters import TimeParameters

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement

plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)      # fontsize of the axes title
plt.rc('axes', labelsize=14.0)      # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)     # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)     # fontsize of the tick labels
plt.rcParams['figure.figsize'] = [9, 6 ] # figure size in inches


@unique
class StatesIsometric(Enum):
    STIMULATION = 0
    L_CE = 1
    NB_STATES = 2


@unique
class StatesIsotonic(Enum):
    STIMULATION = 0
    L_CE = 1
    LOAD_POS = 2
    LOAD_SPEED = 3
    NB_STATES = 4


def find_ce_stretch_iso(muscle_sys, ce_stretch, time_param, stimulation=1, error_max=0.01):
    """Finds the total stretch [m] to apply to obtain a certain ce_stretch(relative) in isometric mode"""

    stretch = 0
    # The stretch is constant for over compression (i.e[0,sth]) hence we wanna monitor this to avoid being
    # in a regime where the computations are biased.
    minimal_ce_stretch = None
    x0 = [0.0, muscle_sys.muscle.L_OPT]

    # Running experiments until we find a stretch that matches the conditions on CE
    while True:
        result = muscle_sys.integrate(x0=x0,
                                      time=time_param.times,
                                      time_step=time_param.t_step,
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
        - handle: handle of the graph. If None title will be used as handle
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
    """Add a new force (force)  on a length (l) to a plot (handle)"""
    plt.figure(handle)
    plt.plot(l, force)
    plt.grid()


def plot_isometric_data(active_force, passive_force, total_force, l_ce, l_slack, l_mtc, handle, l_opt=None):
    """
    Plots the three forces (active, passive and total) for each three graphs (function of l_ce, l_mtc and l_ce).
    Adds a vertical line @l_opt on the l_ce graph.
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

    add_l_opt_marker(handle[0], l_opt)  # Add the l_opt marker on the l_ce graph


def add_l_opt_marker(handle, l_opt=None):
    """Add a vertical labeled line on the graph  @l_opt (uses the default l_opt if l_opt is None)"""
    if l_opt is None:
        muscle_parameters = MuscleParameters()
        l_opt = muscle_parameters.l_opt
    plt.figure(handle)
    plt.axvline(x=l_opt, color="k")
    plt.text(l_opt+0.001, 2000, "L_opt", rotation=-90, color="k")


def get_common_axis_limits(handles):
    """ Finds the axis limits to use in order to merge all graph (specified in the handels lists) in a single graph
    without cutting any part of any curves. """

    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    # Test limits for all axis on all graphs
    for handle in handles:
        plt.figure(handle)
        x_bot, x_top = plt.xlim()
        y_bot, y_top = plt.ylim()
        x_min = min(x_min, x_bot)
        x_max = max(x_max, x_top)
        y_min = min(y_min, y_bot)
        y_max = max(y_max, y_top)
    return x_min, x_max, y_min, y_max


def isometric_experiment(muscle_stimulation=1., ce_stretch_max=1.5, ce_stretch_min=0.5, nb_pts=1000,
                         time_param=TimeParameters(), l_opt=None):
    """ Runs a experiments in isometric mode for multiple stretches, returns the results
    Parameters:
        - muscle_stimulation: applied stimulation.
        - ce_stretch_max: the maximal stretch to apply to the contractile element.
        - ce_stretch_min: the minimal stretch to apply to the contractile element.
        - nb_pts: the number of times the experiment should be ran between the min and max stretch.
         (i.e number of points in the results)
        - time_param: A TimeParameters object to pass the intended time parameters for every experiment.
        - l_opt: The optimal length of the contractile element. If None the default one is taken
    Returns every parameters for the experiments.
    """

    # System definition
    parameters = MuscleParameters()
    if l_opt is not None:
        parameters.l_opt = l_opt
    muscle = Muscle(parameters)
    sys = IsometricMuscleSystem()
    sys.add_muscle(muscle)

    # Experiment parameters
    muscle_stretch_max = find_ce_stretch_iso(sys, ce_stretch_max, time_param)
    muscle_stretch_min = find_ce_stretch_iso(sys, ce_stretch_min, time_param)
    stretches = np.arange(muscle_stretch_min, muscle_stretch_max, muscle_stretch_max / nb_pts)
    x0 = [0]*StatesIsometric.NB_STATES.value
    x0[StatesIsometric.STIMULATION.value] = 0
    x0[StatesIsometric.L_CE.value] = sys.muscle.L_OPT

    # Containers
    active_force = []
    passive_force = []
    total_force = []
    l_ce = []
    l_slack = []
    l_mtc = []

    # Experiences
    pylog.info("Running isometric experiments for stretches (this might take a while)...")
    for stretch in tqdm(stretches):
        result = sys.integrate(x0=x0,
                               time=time_param.times,
                               time_step=time_param.t_step,
                               stimulation=muscle_stimulation,
                               muscle_length=stretch)
        active_force.append(result.active_force[-1])
        passive_force.append(result.passive_force[-1])
        total_force.append(result.tendon_force[-1])
        l_ce.append(result.l_ce[-1])
        l_mtc.append(result.l_mtc[-1])
        l_slack.append(result.l_mtc[-1]-result.l_ce[-1])

    return active_force, passive_force, total_force, l_ce, l_slack, l_mtc


def isotonic_experiment(muscle_stimulation, loads, time_param=TimeParameters()):

    # Muscle
    muscle_parameters = MuscleParameters()
    muscle = Muscle(muscle_parameters)

    # Initial conditions
    x0 = [0] * StatesIsotonic.NB_STATES.value
    x0[StatesIsotonic.STIMULATION.value] = 0.
    x0[StatesIsotonic.L_CE.value] = muscle.L_OPT
    x0[StatesIsotonic.LOAD_POS.value] = muscle.L_OPT + muscle.L_SLACK
    x0[StatesIsotonic.LOAD_SPEED.value] = 0.

    # Containers
    v_ce = []
    tendon_force = []

    # Integration
    pylog.info("Running the experiments (this might take a while)...")
    for load in tqdm(loads):

        # New load definition
        mass_parameters = MassParameters()
        mass_parameters.mass = load
        mass = Mass(mass_parameters)

        # System definition
        sys = IsotonicMuscleSystem()
        sys.add_muscle(muscle)
        sys.add_mass(mass)

        result = sys.integrate(x0=x0,
                               time=time_param.times,
                               time_step=time_param.t_step,
                               time_stabilize=time_param.t_stabilize,
                               stimulation=muscle_stimulation,
                               load=load)

        # Result processing
        if result.l_mtc[-1] > x0[StatesIsotonic.LOAD_POS.value]:
            # Extension
            index = result.v_ce.argmax()
            v_ce.append(result.v_ce.max())
            tendon_force.append(result.tendon_force[index])
        else:
            # Contraction
            index = result.v_ce.argmin()
            v_ce.append(result.v_ce.min())
            tendon_force.append(result.tendon_force[index])

    return v_ce, tendon_force


def exercise1a(time_param):
    """ Runs and plot the length-force relationship for a muscle in isometric conditions."""
    pylog.info("Part a")

    # Parameters
    muscle_stimulation = 1.
    ce_stretch_max = 1.5
    ce_stretch_min = 0.5
    nb_pts = 1000
    handles = ["1_a_lce_vs_str", "1_a_lsl_vs_str", "1_a_lmtc_vs_str"]

    # Experiment
    active_force, passive_force, total_force, l_ce, l_slack, l_mtc = isometric_experiment(muscle_stimulation,
                                                                                          ce_stretch_max,
                                                                                          ce_stretch_min,
                                                                                          nb_pts,
                                                                                          time_param,)
    # Plotting
    plot_isometric_data(active_force, passive_force, total_force, l_ce, l_slack, l_mtc, handle=handles)


def exercise1b(time_param):
    """Runs and plot the length-force relationship for a muscle in isometric conditions and various activations
    amplitudes."""
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

    # region Experiences (expect it to take some time)
    for stim in stimulations:
        # Running the integration
        pylog.info("Running isometric experiences for stimulation={}".format(stim))
        active_force, passive_force, total_force, l_ce, l_slack, l_mtc = isometric_experiment(stim,
                                                                                              ce_stretch_max,
                                                                                              ce_stretch_min,
                                                                                              nb_pts,
                                                                                              time_param)
        # adding active force
        lengths = [l_ce, l_slack, l_mtc]
        force_legends.append("Active force, stimulation={}".format(stim))
        # Plots
        for handle, length in zip(figure_handles, lengths):
            add_force_to_plot(handle, active_force, length)

    # Adding passive force only once (it isn't influenced by stimulation)
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
    """ Plots the force-length relationship for two muscle with different l_ce in isometric conditions."""
    pylog.info("part C")

    # region Parameters
    l_opt_long = 0.5
    l_opt_small = 0.25
    stim = 1.
    ce_stretch_max = 1.5
    ce_stretch_min = 0.5
    nb_pts = 1000
    # endregion

    # region Experiments
    pylog.info("Running isometric experiment for long ce")
    active_force_long, passive_force_long, total_force_long, l_ce_long, l_slack_long, l_mtc_long = \
        isometric_experiment(stim, ce_stretch_max, ce_stretch_min, nb_pts, time_param, l_opt_long)
    pylog.info("Running isometric experiment for small ce")
    active_force_small, passive_force_small, total_force_small, l_ce_small, l_slack_small, l_mtc_small = \
        isometric_experiment(stim, ce_stretch_max, ce_stretch_min, nb_pts, time_param, l_opt_small)
    # endregion

    # region Plots
    # Plots parameters
    handles_long = ["1_c_lce_iso_long", "1_c_lsl_iso_long", "1_c_lmtc_long"]
    handles_small = ["1_c_lce_iso_small", "1_c_lsl_iso_small", "1_c_lmtc_small"]
    handles_merged = ["1_c_lce_iso_merged", "1_c_lsl_iso_merged", "1_c_lmtc_merged"]
    merged_legends = ["Active force, l_opt=0.5", "Passive force, l_opt=0.5", "Total force, l_opt=0.5", "_nolegend_",
                      "Active force, l_opt=0.25", "Passive force, l_opt=0.25", "Total force, l_opt=0.25"]

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

    # Axis equalisation for merged plots
    for handle_long, handle_small, handle_merged in zip(handles_long, handles_small, handles_merged):
        x_min, x_max, y_min, y_max = get_common_axis_limits([handle_small, handle_long])
        plt.figure(handle_merged)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid()
        plt.legend(merged_legends, prop={'size': 8})
    # endregion


def exercise1d(time_param):
    """ Plots the v_ce-force relationship by running experiment with multiple weights."""
    pylog.info("Exercise 1d")

    # Parameters
    load_min = 20
    load_max = 300
    nb_loads = 1000
    load = np.linspace(load_min, load_max, num=nb_loads)
    muscle_stimulation = 1.

    # Experiment
    v_ce, tendon_force = isotonic_experiment(loads=load, muscle_stimulation=muscle_stimulation, time_param=time_param)

    # Plotting
    plt.figure('1_d_isotonic_load_var')
    plt.plot(v_ce, tendon_force)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Contractile element velocity [m/s]')
    plt.ylabel('Tendon force [N]')
    plt.axvline(x=0, color="k")
    plt.text(0.001, 750, "Equilibrium (v=0)", rotation=-90, color="k")
    plt.grid()


def exercise1e(time_param):
    """ Plots the v_ce-force relationship by running experiment with multiple weights and multiple stimlations."""
    pylog.info("Exercise 1e")

    # Parameters
    load_min = 20
    load_max = 300
    nb_loads = 1000
    load = np.linspace(load_min, load_max, num=nb_loads)

    stimulation_min = 0
    stimulation_max = 1
    nb_stimulation = 10
    stimulations = np.linspace(stimulation_min, stimulation_max, num=nb_stimulation)

    # Containers for 3D plots
    v_ce_3d = np.zeros(shape=(nb_stimulation, nb_loads))
    tendon_force_3d = np.zeros(shape=(nb_stimulation, nb_loads))
    stimulations_3d = np.ones(shape=(nb_stimulation, nb_loads))
    # Experiment
    for i, stimulation in enumerate(stimulations):
        v_ce, tendon_force = isotonic_experiment(loads=load, muscle_stimulation=stimulation, time_param=time_param)

        v_ce_3d[i, :] = v_ce
        tendon_force_3d[i, :] = tendon_force
        stimulations_3d[i, :] *= stimulation

        # Plotting
        plt.figure('1_d_isotonic_stim_var')
        plt.plot(v_ce, tendon_force, label='stimulation=%.1f'%stimulation)
        plt.title('Isotonic muscle experiment')
        plt.xlabel('Contractile element velocity [m/s]')
        plt.ylabel('Tendon force [N]')

    plt.legend()
    plt.grid()

    # 3d plot
    fig = plt.figure('1_d_isotonic_3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(stimulations_3d, v_ce_3d, tendon_force_3d, cmap=cm.coolwarm)
    ax.set_xlabel('Stimulation')
    ax.set_ylabel('Contractile element velocity [m/s]')
    ax.set_zlabel('Tendon force [N]')


def exercise1():
    plt.close("all")
    pylog.info("Start exercise 1")
    time_param = TimeParameters(time_start=0.0, time_stop=0.2, time_step=0.001, time_stabilize=0.2)

    exercise1a(time_param)
    # exercise1b(time_param)
    # exercise1c(time_param)
    # time_param.t_stop = 0.3  # change time parameters for the second part
    # exercise1d(time_param)
    # exercise1e(time_param)

    if DEFAULT["save_figures"]:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
    plt.show()


if __name__ == '__main__':
    parse_args()
    exercise1()

