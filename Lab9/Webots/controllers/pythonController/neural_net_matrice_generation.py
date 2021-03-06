""" When imported writes new phase bias and weights matrices in save_folder"""

import numpy as np
from math import inf

from simulation_parameters import SimulationParameters


# Which leg is coupled rto which body oscillator. Had to be hard coded due to incoherent numbering.
leg_to_body_coupling = [[1, 2, 3, 4, 5],
                        [11, 12, 13, 14, 15],
                        [6, 7, 8, 9],
                        [16, 17, 18, 19]]

leg_to_leg_coupling = [[21, 22],
                       [20, 23],
                       [20, 23],
                       [21, 22]]

save_folder = 'Arrays\\'
coupling_weights_file = save_folder + 'coupling_weights.csv'
phase_bias_file = save_folder + 'phase_bias.csv'


class FileGenerator:
    
    def __init__(self, parameters=SimulationParameters()):

        # Robot definitions
        self.parameters = parameters
        self.n_oscillators_body = parameters.n_body_joints * 2
        self.n_oscillators_legs = parameters.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs

        # Tables
        self.coupling_weights = np.zeros([self.n_oscillators, self.n_oscillators])

        # Unconnected phases are represented by inf for easier checks
        self.phase_bias = np.full([self.n_oscillators, self.n_oscillators], inf)
    
    def generate_weights_file(self):
    
        # region Couple body oscillators
        for i in range(self.n_oscillators_body):
            # Neighbour coupling
            if i != (self.n_oscillators_body / 2) - 1 and i < (self.n_oscillators_body - 1):
                self.coupling_weights[i][i + 1] = self.parameters.weak_coupling
                self.coupling_weights[i + 1][i] = self.parameters.weak_coupling
            # Anti-phase coupling
            if i < int(self.n_oscillators_body / 2):
                self.coupling_weights[i][i + int(self.n_oscillators_body / 2)] = self.parameters.weak_coupling
                self.coupling_weights[i + int(self.n_oscillators_body / 2)][i] = self.parameters.weak_coupling
        # endregion

        # region Limb coupling
        for i in range(self.n_oscillators_legs):
            for j in leg_to_leg_coupling[i][:]:
                self.coupling_weights[self.n_oscillators_body + i][j] = self.parameters.weak_coupling
        # endregion

        # region Limb to body coupling
        for i in range(self.n_oscillators_legs):
            for j in leg_to_body_coupling[i][:]:
                self.coupling_weights[j][self.n_oscillators_body + i] = self.parameters.strong_coupling
        # endregion

        # Save as CSV file
        np.savetxt(coupling_weights_file, self.coupling_weights, delimiter=',')
    
    def generate_phase_bias_file(self):

        # Phase definitions
        in_phase = 0
        in_anti_phase = np.pi
        phase_lag = 2*np.pi/(self.n_oscillators_body/2)

        # region Couple body oscillators
        for i in range(self.n_oscillators_body):
            # Neighbour coupling
            if i != (self.n_oscillators_body / 2) - 1 and i < (self.n_oscillators_body - 1):
                self.phase_bias[i][i + 1] = phase_lag
                self.phase_bias[i + 1][i] = -phase_lag
            # Anti-phase coupling
            if i < int(self.n_oscillators_body / 2):
                self.phase_bias[i][i + int(self.n_oscillators_body / 2)] = in_anti_phase
                self.phase_bias[i + int(self.n_oscillators_body / 2)][i] = in_anti_phase
        # endregion

        # region Limb coupling
        for i in range(self.n_oscillators_legs):
            for j in leg_to_leg_coupling[i][:]:
                self.phase_bias[self.n_oscillators_body + i][j] = in_anti_phase
        # endregion

        # region Limb to body coupling
        for i in range(self.n_oscillators_legs):
            for j in leg_to_body_coupling[i][:]:
                self.phase_bias[j][self.n_oscillators_body + i] = in_anti_phase
        # endregion

        # Save as CSV file
        np.savetxt(phase_bias_file, self.phase_bias, delimiter=',')


file_generator = FileGenerator()
file_generator.generate_weights_file()
file_generator.generate_phase_bias_file()
