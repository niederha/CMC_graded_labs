"""Run network without Webots"""

import time
import numpy as np
import matplotlib.pyplot as plt
import cmc_pylog as pylog
from network import SalamanderNetwork
from save_figures import save_figures
from parse_args import save_plots
from simulation_parameters import SimulationParameters
import plot_results as plt_res


def run_network(duration, update=False, drive=1., gait="Walking"):
    """Run network without Webots and plot results"""
    # Simulation setup
    timestep = 5e-3
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag=None,
        turn=None,
    )
    network = SalamanderNetwork(timestep, parameters)
    osc_left = np.arange(10)
    osc_right = np.arange(10, 20)
    osc_legs = np.arange(20, 24)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases)
    ])
    phases_log[0, :] = network.state.phases
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes)
    ])
    amplitudes_log[0, :] = network.state.amplitudes
    freqs_log = np.zeros([
        n_iterations,
        len(network.parameters.freqs)
    ])
    freqs_log[0, :] = network.parameters.freqs
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output())
    ])
    outputs_log[0, :] = network.get_motor_position_output()

    # Run network ODE and log data
    tic = time.time()
    for i, _ in enumerate(times[1:]):
        if update:
            network.parameters.update(
                SimulationParameters(
                    # amplitude_gradient=None,
                    # phase_lag=None
                )
            )
        network.step()
        phases_log[i+1, :] = network.state.phases
        amplitudes_log[i+1, :] = network.state.amplitudes
        outputs_log[i+1, :] = network.get_motor_position_output()
        freqs_log[i+1, :] = network.parameters.freqs
    toc = time.time()

    # Network performance
    pylog.info("Time to run simulation for {} steps: {} [s]".format(
        n_iterations,
        toc - tic
    ))

    plt_res.plot_body_joints(times, outputs_log, gait + ' body joints')
    plt_res.plot_leg_joints(times, outputs_log, gait + ' leg joints')

    
def main(save):
    """Main"""

    run_network(duration=25, drive=1., gait='Walking')
    #run_network(duration=15, drive=4., gait='Swimming')

    # Show plots
    if save:
        save_figures()
    plt.show()


if __name__ == '__main__':
    main(save=False)

