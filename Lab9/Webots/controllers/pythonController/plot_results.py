"""Plot results"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_body_joints(time, joint_angles, variable='body joint angle'):
    nb_legs = 4
    nb_body = joint_angles.shape[1]-nb_legs
    plt.figure(variable.replace(" ", "_"))
    offset = joint_angles[:nb_body].max()-joint_angles[:nb_body].min()
    for body_joint_index in range(nb_body):
        plt.plot(time, joint_angles[:, body_joint_index]+0.75*body_joint_index*offset,
                 label="body joint " + str(body_joint_index))
    plt.grid()
    plt.legend()
    plt.title(variable + " evolution")


def plot_leg_joints(time, joint_angles, variable='leg joint angle'):
    nb_legs = 4
    nb_body = joint_angles.shape[1]-nb_legs
    plt.figure(variable.replace(" ", "_"))
    offset = joint_angles[nb_body:].max()-joint_angles[nb_body:].min()

    for leg_joint_index in range(nb_legs):
        plt.plot(time, joint_angles[:, nb_body+leg_joint_index]+0.75*leg_joint_index*offset,
                 label="leg joint " + str(leg_joint_index))
    plt.grid()
    plt.legend()
    plt.title(variable + " evolution")


def plot_3D_variable(times, variable_log, variable='joint angle'):

    nb_joints = variable_log.shape[1]

    # 3D data containers
    times_3d = np.zeros_like(variable_log)
    joint_number_3d = np.zeros_like(variable_log)

    for i in range(nb_joints):
        times_3d[:, i] = times

    for i in range(len(times)):
        joint_number_3d[i, :] = np.arange(nb_joints)

    # 3D plots
    fig = plt.figure(variable.replace(" ", "_")+"_plot")
    plt.title(variable)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(times_3d, joint_number_3d, variable_log, cmap=cm.coolwarm)
    ax.set_xlabel('time')
    ax.set_ylabel('joint number')
    ax.set_zlabel(variable)


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def main(plot=True):
    """Main"""
    # Load data
    with np.load('logs/example/simulation_0.npz') as data:
        timestep = float(data["timestep"])
        amplitude = data["amplitudes"]
        phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    # Plot data
    plt.figure("Positions")
    plot_positions(times, link_data)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

