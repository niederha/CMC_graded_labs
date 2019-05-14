"""Plot results"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots

from mpl_toolkits.mplot3d import Axes3D  # Module for 3D plots
from matplotlib import cm  # Module for color maps


def plot_body_joints(time, joint_angles, variable='body joint angle',title='Spine angle evolution'):
    """ Extracts and plots motor output for body joints"""
    nb_legs = 4
    nb_body = joint_angles.shape[1]-nb_legs
    plt.figure(variable.replace(" ", "_"))
    offset = joint_angles[:, :nb_body].max()-joint_angles[:, :nb_body].min()
    for body_joint_index in range(nb_body):
        plt.plot(time, joint_angles[:, body_joint_index]+1.*(nb_body-body_joint_index-1)*offset,
                 label="body joint " + str(body_joint_index))
    plt.grid()
    plt.legend()
    plt.title(title)


def plot_leg_joints(time, joint_angles, variable='leg joint angle'):
    """ Extracts and plots motor output for body joints"""
    nb_legs = 4
    nb_body = joint_angles.shape[1]-nb_legs
    plt.figure(variable.replace(" ", "_"))

    # Wrap up legs output:
    joint_angles[:, nb_body:] %= 2 * np.pi

    offset = joint_angles[:, nb_body:].max()-joint_angles[:, nb_body:].min()
    if offset == 0:
        offset = 0.5

    for leg_joint_index in range(nb_legs):
        plt.plot(time, joint_angles[:, nb_body+leg_joint_index]+1.1*(nb_legs-1-leg_joint_index)*offset,
                 label="leg joint " + str(leg_joint_index))
    plt.grid()
    plt.legend()
    plt.title(variable + " evolution")


def plot_3d_variable(times, variable_log, variable_name):
    """ Plots a variable as a 3D surface depending on joint_number (Y axis) and time(X axis)"""
    nb_joints = variable_log.shape[1]

    # 3D data containers
    times_3d = np.zeros_like(variable_log)
    joint_number_3d = np.zeros_like(variable_log)

    for i in range(nb_joints):
        times_3d[:, i] = times

    for i in range(len(times)):
        joint_number_3d[i, :] = np.arange(nb_joints)

    # 3D plots
    fig = plt.figure(variable_name.replace(" ", "_")+"_plot")
    plt.title(variable_name)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(times_3d, joint_number_3d, variable_log, cmap=cm.coolwarm)
    ax.set_xlabel('time')
    ax.set_ylabel('joint number')
    ax.set_zlabel(variable_name)


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)
    

def plot_trajectory(link_data,label):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 2], label=label)
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.title('GPS trajectory '+ label)


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

def exercise_9c_plots():
    
    link_vel_list = []
    joint_vel_list = []
    joint_torque_list = []
    rhead_list = [] 
    rtail_list = [] 
    for sim in range(0,25):
        with np.load('logs/exercise_9c/simulation_{0}.npz' .format(sim)) as data:
            timestep = float(data["timestep"])
            link_data = data["links"][:, 0, :]
            joints_data = data["joints"]
            rhead = data["RHead"]
            rtail = data["RTail"]
        #times = np.arange(0, timestep*np.shape(link_data)[0], timestep)


        pos = link_data
        vel = np.diff(pos, axis=0, prepend=0)/ timestep

        link_vel_list.append(np.linalg.norm(np.mean(vel,axis = 0)))        
        
        rhead_list.append(rhead)
        rtail_list.append(rtail)
        
        joint_velocity = joints_data[:,:,1]
        joint_vel_list.append(joint_velocity)
    
        joint_torque = joints_data[:,:,3]
        joint_torque_list.append(joint_torque)
        
    """Plot the velocity 3d graph"""
    
    fig = plt.figure("exercise_9c_velocity_plot")
    ax = fig.add_subplot(111, projection='3d')
    
    plt.title("Velocity vs RHead/RTail")
    
    X = np.array(rhead_list)
    Y = np.array(rtail_list)
    X = X.reshape((5,5))
    Y = Y.reshape((5,5))
    
    Z = np.array(link_vel_list)
    
    Z = Z.reshape((5,5))
    
    ax.plot_surface(X,Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('RHead')
    ax.set_ylabel('RTail')
    ax.set_zlabel('Velocity [m/s]')
    
    """Plot the energy graph"""
    
    torque = np.array(joint_torque_list)
    velocity = np.array(joint_vel_list)

    energy = torque*velocity*timestep

    energy = np.sum(energy,axis = 1)

    energy = np.sum(energy,axis = 1)

    
    fig = plt.figure("exercise_9c_energy_plot")
    ax = fig.add_subplot(111, projection='3d')
    
    plt.title("Energy vs RHead/RTail")
    
    X = np.array(rhead_list)
    Y = np.array(rtail_list)
    X = X.reshape((5,5))
    Y = Y.reshape((5,5))
    
    Z = energy
    
    Z = Z.reshape((5,5))
    
    ax.plot_surface(X,Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('RHead')
    ax.set_ylabel('RTail')
    ax.set_zlabel('Energy [J]')
    
def exercise_9d1_plots():
    
    with np.load('logs/exercise_9d1/simulation_0.npz') as data:
        timestep = float(data["timestep"])
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]

    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    pos = link_data      
    
    """Plot gps trajectory"""
    
    plt.figure("exercise_9d1_trajectory_plot")
    plot_trajectory(pos,'Turning')

    """Plot spine angles"""
    joint_angles = joints_data[:,:,0]
    
    plot_body_joints(times[:10000], joint_angles[:10000], variable='exercise_9d1_spine_angles_plot',title='Spine angle evolution turning')

def exercise_9d2_plots():
    
    with np.load('logs/exercise_9d2/simulation_0.npz') as data:
        timestep = float(data["timestep"])
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]

    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    pos = link_data      
    
    """Plot gps trajectory"""
    
    plt.figure("exercise_9d2_trajectory_plot")
    plot_trajectory(pos,'Backwards')

    """Plot spine angles"""
    joint_angles = joints_data[:,:,0]
    
    plot_body_joints(times[:10000], joint_angles[:10000], variable='exercise_9d2_spine_angles_plot',title='Spine angle evolution backwards')


def main(save=False):
    """Main"""
    # Load data
    exercise_9c_plots()
    exercise_9d1_plots()
    exercise_9d2_plots()
    # Show plots
    if save:
        save_figures()
    plt.show()


if __name__ == '__main__':
    main(save=False)

