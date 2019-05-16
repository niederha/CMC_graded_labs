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


def plot_body_joints(time, joint_angles, file_name='body joint angle', title='Spine angle evolution', offset_mult=1.1,
                     gait='swimming'):
    """ Extracts and plots motor output for body joints"""
    nb_legs = 4
    nb_body = joint_angles.shape[1]-nb_legs
    plt.figure(file_name.replace(" ", "_"))

    offset = joint_angles[:, :nb_body].max()-joint_angles[:, :nb_body].min()

    for body_joint_index in range(nb_body):
        if gait == 'walking':
            offset_add_head = 1.5
            offset_add_upper_bod = 1.5
            if body_joint_index == 0:
                plt.plot(time,
                         joint_angles[:, body_joint_index]
                         + offset_mult * (nb_body-body_joint_index-1) * offset
                         + offset_add_head * offset + offset_add_upper_bod * offset,
                         label=f'head joint {body_joint_index}')
            elif body_joint_index < 6:
                plt.plot(time,
                         joint_angles[:, body_joint_index]
                         + offset_mult * (nb_body - body_joint_index - 1) * offset
                         + offset_add_upper_bod * offset,
                         label=f'upper body joint {body_joint_index}')
            else:
                plt.plot(time,
                         joint_angles[:, body_joint_index]
                         + offset_mult * (nb_body - body_joint_index - 1) * offset,
                         label=f'lower body joint {body_joint_index}')
        else:
            plt.plot(time,
                     joint_angles[:, body_joint_index]
                     + offset_mult * (nb_body - body_joint_index - 1) * offset,
                     label=f'body joint {body_joint_index}')

    plt.grid()
    plt.legend()
    plt.title(title)


def plot_leg_joints(time, joint_angles, file_name='leg joint angle', title='Limb angle evolution', offset_mult=1.1):
    """ Extracts and plots motor output for body joints"""
    nb_legs = 4
    nb_body = joint_angles.shape[1]-nb_legs
    plt.figure(file_name.replace(" ", "_"))

    # Wrap up legs output:
    joint_angles[:, nb_body:] %= 2 * np.pi

    offset = joint_angles[:, nb_body:].max()-joint_angles[:, nb_body:].min()
    if offset == 0:
        offset = 0.5

    for leg_joint_index in range(nb_legs):
        plt.plot(time, joint_angles[:, nb_body+leg_joint_index]+offset_mult*(nb_legs-1-leg_joint_index)*offset,
                 label="leg joint " + str(leg_joint_index))
    plt.grid()
    plt.legend()
    plt.title(title)


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

def exercise_9f1_plots():
    
    link_vel_list = []
    phase_lag_list = []
    
    for sim in range(0,15):
        with np.load('logs/exercise_9f1/simulation_{0}.npz' .format(sim)) as data:
            timestep = float(data["timestep"])
            link_data = data["links"][:, 0, :]
            phase_lag = data["phase_lag_body_limb"]
            
        times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
        phase_lag_list.append(phase_lag)

        pos = link_data
        vel = np.diff(pos, axis=0, prepend=0)/ timestep
#        meanvel = np.linalg.norm(pos[-1] -pos[0])/times[-1]
#        plt.figure()
#        plt.plot(vel[25:])
#        vel[vel>0.3] = 0
#        vel[vel<-0.3] = 0
#        print(vel[:,0])
        link_vel_list.append((np.mean(vel[-2400:,0],axis = 0)))  
#        link_vel_list.append(meanvel)
        
    """Plot the velocity 2d graph"""
    
    plt.figure("exercise_9f1_velocity_plot")
    
    plt.title("Velocity vs phase lag body limb")
    
    X = np.array(phase_lag_list)
    
    Y = np.array(link_vel_list)
    print(np.argmax(Y))
    print(X[np.argmax(Y)])
    plt.plot(X,Y,label='Velocity')
    plt.legend()
    plt.xlabel('phase_lag')
    plt.ylabel('Velocity [m/s]')
    
def exercise_9f2_plots():
    
    link_vel_list = []
    amplitudes_list = []
    
    for sim in range(0,15):
        with np.load('logs/exercise_9f2/simulation_{0}.npz' .format(sim)) as data:
            timestep = float(data["timestep"])
            link_data = data["links"][:, 0, :]
            amplitude = data["amplitudesLimb"]
            
        #times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
        amplitudes_list.append(amplitude)

        pos = link_data
        vel = np.diff(pos, axis=0, prepend=0)/ timestep
#        plt.figure()
#        plt.plot(vel[25:])
#        vel[vel>0.3] = 0
#        vel[vel<-0.3] = 0

        link_vel_list.append(np.linalg.norm(np.mean(vel[-2400:,0],axis = 0)))        
        
    """Plot the velocity 2d graph"""
    
    plt.figure("exercise_9f2_velocity_plot")
    
    plt.title("Velocity vs nominal Radius")
    
    X = np.array(amplitudes_list)
    
    Y = np.array(link_vel_list)
    
    plt.plot(X,Y,label='Velocity')
    plt.legend()
    plt.xlabel('nominal R')
    plt.ylabel('Velocity [m/s]')
    
def exercise_9f_walking_plots():
    
    with np.load('logs/exercise_9f_walking/simulation_0.npz') as data:
        timestep = float(data["timestep"])
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]

    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    """Plot spine angles"""
    joint_angles = joints_data[:,:,0]
    plot_body_joints(times, joint_angles, variable='exercise_9f_walking_spine_angles_plot',title='Spine angle evolution walking')
    plot_leg_joints(times, joint_angles, variable='exercise_9f_walking_limb_angles_plot',title='Limb angle evolution walking')
    #print(np.argmin(joint_angles[-600:-100,:10], axis=0))
    time_lags = times[-750+np.argmin(joint_angles[-750:-250,:10], axis=0)]
    time_lags_max = times[-750+np.argmax(joint_angles[-750:-250,:10], axis=0)]
    total_time_lag = time_lags[-1]-time_lags[0]
    period = (time_lags_max[0]-time_lags[0])*2
    print(period)
    lag = total_time_lag/10/period
    print(lag*2*np.pi)
    
        
def exercise_9g_plots():
    
    with np.load('logs/exercise_9g/simulation_0.npz') as data:
        timestep = float(data["timestep"])
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]

    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    pos = link_data      
    
    """Plot gps trajectory"""
    
    plt.figure("exercise_9g_trajectory_plot")
    plot_trajectory(pos,'Transition')

    """Plot spine angles"""
    joint_angles = joints_data[:,:,0]
    plot_body_joints(times, joint_angles, variable='exercise_9g_spine_angles_plot',title='Spine angle evolution g/w/g transition')
    plt.xlabel('time [s]')
    plt.axvline(x=14.5, color="k") 
    plt.text(14.5+0.001, -1, "Transition", rotation=-90, color="k")
    plt.axvline(x=18, color="k") 
    plt.text(18-2, -1, "Transition", rotation=-90, color="k")
    plot_leg_joints(times, joint_angles, variable='exercise_9g_limb_angles_plot',title='Limb angle evolution g/w/g transition')
    plt.xlabel('time [s]')
    plt.axvline(x=14.5, color="k") 
    plt.text(14.5+0.001, 4, "Transition", rotation=-90, color="k")
    plt.axvline(x=18, color="k") 
    plt.text(18-2, 4, "Transition", rotation=-90, color="k")
    plot_body_joints(pos[:int(14/timestep),0], joint_angles[:int(14/timestep)], variable='exercise_9g_GW_spine_angles_plot',title='Spine angle evolution ground/water transition')
    plt.xlabel('x [m]')
    plot_leg_joints(pos[:int(14/timestep),0], joint_angles[:int(14/timestep)], variable='exercise_9g_GW_limb_angles_plot',title='Limb angle evolution ground/water transition')
    plt.xlabel('x [m]')
    plot_body_joints(pos[int(18/timestep):,0], joint_angles[int(18/timestep):], variable='exercise_9g_WG_spine_angles_plot',title='Spine angle evolution water/ground transition')
    plt.xlabel('x [m]')
    plt.gca().invert_xaxis()
    plot_leg_joints(pos[int(18/timestep):,0], joint_angles[int(18/timestep):], variable='exercise_9g_WG_limb_angles_plot',title='Limb angle evolution water/ground transition')
    plt.xlabel('x [m]')
    plt.gca().invert_xaxis()

def main(save=False):
    """Main"""
    # Load data
    Exercise_9b_plot_gridsearch_speed()
    Exercise_9b_plot_gridsearch_energy()
    ###Exercise_9b_plot_gridsearch_performance()
    
    #exercise_9c_plots()
    
    #exercise_9d1_plots()
    #exercise_9d2_plots()
    
    #exercise_9f_walking_plots()
    #exercise_9f1_plots()
    #exercise_9f2_plots()
    
    #exercise_9g_plots()
    
    # Show plots
    if save:
        save_figures()
    plt.show()


if __name__ == '__main__':
    main(save=False)

