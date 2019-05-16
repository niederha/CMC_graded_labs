"""Plot results"""

import numpy as np
from scipy.interpolate import griddata
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots

from mpl_toolkits.mplot3d import Axes3D  # Module for 3D plots
from matplotlib import cm  # Module for color maps

plt.close("all")

def plot_body_joints(time, joint_angles, variable='body joint angle'):
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
    plt.title(variable + " evolution")


def plot_leg_joints(time, joint_angles, variable='leg joint angle'):
    """ Extracts and plots motor output for body joints"""
    nb_legs = 4
    nb_body = joint_angles.shape[1]-nb_legs
    plt.figure(variable.replace(" ", "_"))

    # Wrap up legs output:
    joint_angles[:, nb_body:] %= 2 * np.pi

    offset = joint_angles[:, nb_body:].max()-joint_angles[:, nb_body:].min()
    if offset == 0:
        offset = 1

    for leg_joint_index in range(nb_legs):
        plt.plot(time, joint_angles[:, nb_body+leg_joint_index]+1.*(nb_legs-1-leg_joint_index)*offset,
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
    
def Exercise_9b_plot_gridsearch_speed():
    meanspeed=[]

    amplitudes=[]
    phase_lags=[]
    
    for i in range(0,300):
        path='logs/exercise_9b_300/simulation_'+str(i)+'.npz'
        with np.load(path) as data:
            timestep = float(data["timestep"])
            position = data["links"][:, 0, :]

            
        xvelocity = np.diff(position[:,0], axis=0) / timestep     
        yvelocity = np.diff(position[:,1], axis=0) / timestep
        meanspeed= np.append(meanspeed,np.mean(xvelocity)+np.mean(yvelocity))

        
    meanspeed=np.reshape(meanspeed,(10,30))  
  
    phase_lags_3d, amplitudes_3d = np.meshgrid(np.linspace(0, 3, 30),np.linspace(0.1,0.5, 10))
    
    

    # 3D plots
    fig = plt.figure("9b_SpeedPhaseLagsAmplitudes")
    plt.title("Speed")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(phase_lags_3d, amplitudes_3d, meanspeed, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel('phase lag [rad]')
    ax.set_ylabel('oscillation amplitude')
    ax.set_zlabel("Robot Speed [m/s]")
    
    
    print(np.where(meanspeed == meanspeed.max()))
    print(np.max(meanspeed))

    
def Exercise_9b_plot_gridsearch_energy():
    energies=[]

    amplitudes=[]
    phase_lags=[]
    
    for i in range(0,300):
        path='logs/exercise_9b_300/simulation_'+str(i)+'.npz'
        with np.load(path) as data:
            timestep = float(data["timestep"])
            velocities = data["joints"][:, :, 1]
            torques = data["joints"][:,:,3]
            
            
        energies = np.append(energies, np.sum(torques*velocities*timestep))
        
    energies=np.reshape(energies,(10,30)) 
    phase_lags_3d, amplitudes_3d = np.meshgrid(np.linspace(0, 3, 30),np.linspace(0.1,0.5, 10))


    # 3D plots
    fig = plt.figure("9b_EnergyPhaseLagsAmplitudes")
    plt.title("Speed")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(phase_lags_3d, amplitudes_3d, energies, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel('Phase lag [rad]')
    ax.set_ylabel('Oscillation amplitude ')
    ax.set_zlabel("Robot Energy consumed [J]")  
    print(np.max(energies))
    print(energies[4,5])
    
def Exercise_9b_plot_gridsearch_performance():
    energies=[]
    meanspeed=[]
    amplitudes=[]
    phase_lags=[]
    performance=[]
    
    for i in range(0,300):
        path='logs/exercise_9b_300/simulation_'+str(i)+'.npz'
        with np.load(path) as data:
            timestep = float(data["timestep"])
            velocities = data["joints"][:, :, 1]
            torques = data["joints"][:,:,3]
            position = data["links"][:, 0, :]
            
        energies = np.append(energies, np.sum(torques*velocities*timestep))
        xvelocity = np.diff(position[:,0], axis=0) / timestep     
        yvelocity = np.diff(position[:,1], axis=0) / timestep
        meanspeed= np.append(meanspeed,np.mean(xvelocity)+np.mean(yvelocity))
        performance=np.append(performance,meanspeed[-1]/energies[-1])
        
    performance=np.reshape(performance,(10,30))         
    
    phase_lags_3d, amplitudes_3d = np.meshgrid(np.linspace(0, 3, 30),np.linspace(0.1,0.5, 10))


    # 3D plots
    fig = plt.figure("9b_PerformacePhaseLagsAmplitudes")
    plt.title("Speed")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(phase_lags_3d, amplitudes_3d, performance, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel('Phase lag [rad]')
    ax.set_ylabel('Oscillation amplitude ')
    ax.set_zlabel("Performance (speed/energy)")  
    print(np.max(energies))
    print(energies[4,5])
    
#def Exercise_9b_plot_gridsearch_wavelength():
#    period=[]
#    position=[]
#    amplitudes=[]
#    phase_lags=[]
#    
#    for i in range(0,25):
#        path='logs/exercise_9b/simulation_'+str(i)+'.npz'
#        with np.load(path) as data:
#            timestep = float(data["timestep"])
#            position_data = data["joints"][:,0,0]
#            #position_data = data["links"][:, 0, :]
#            phase_lag_data = data["phase_lag"]
#            amplitudes_data = data["amplitudes"]
#            frequency = data["frequency"]
#        #print(position_data.shape) 
#        plt.figure()
#        plt.plot(position_data)
#        peakind = signal.find_peaks_cwt(-position_data, np.array([100]), min_snr=1)
#        peakind = np.array(peakind)
#        
#        period=np.append(period,(peakind[-2]-peakind[-3])*timestep) 
#    print(frequency)


def main(save=True):
    """Main"""
    # Load data
    with np.load('logs/exercise_9b/simulation_0.npz') as data:
        timestep = float(data["timestep"])
        amplitude = data["amplitudes"]
        phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    #print(amplitude.shape)
    #print(phase_lag.shape)
    #print(link_data.shape)
    #print(joints_data.shape)

    
    # Plot data
    #plt.figure("Positions")
    #plot_positions(times, link_data)

    plt.figure("Joints")
    plot_positions(times, joints_data[:,1,:3])

    #plt.figure("Trajectory")
    #plot_trajectory(link_data)
    
    #plt.figure("2dPlot")
    #plot_2d(link_data, ['x','y','z'])
    Exercise_9b_plot_gridsearch_speed()
    Exercise_9b_plot_gridsearch_energy()
    Exercise_9b_plot_gridsearch_performance()
    #Exercise_9b_plot_gridsearch_wavelength()
    # Show plots
    if save:
        save_figures()
    plt.show()


if __name__ == '__main__':
    main(save=True)

