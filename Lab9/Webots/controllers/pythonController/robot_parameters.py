"""Robot parameters"""

import numpy as np
import neural_net_matrice_generation as nnmg
import cmc_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.mlr_drive = parameters.mlr_drive
        self.phase_lag = parameters.phase_lag
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""

        # Unsaturated body
        if parameters.body_saturation[0] <= parameters.mlr_drive <= parameters.body_saturation[1]:
            self.freqs[:self.n_oscillators_body] = parameters.c_freq_body[0] + \
                                              parameters.mlr_drive * parameters.c_freq_body[1]

        # Saturated body
        else:
            self.freqs[:self.n_oscillators_body] = parameters.v_sat

        # Unsaturated limbs
        if parameters.limb_saturation[0] <= parameters.mlr_drive <= parameters.limb_saturation[1]:
            self.freqs[self.n_oscillators_body:] = parameters.c_freq_limb[0] + \
                                              parameters.mlr_drive * parameters.c_freq_limb[1]

        # Saturated limbs
        else:
            self.freqs[self.n_oscillators_body:] = parameters.v_sat

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        self.coupling_weights = np.genfromtxt(nnmg.coupling_weights_file, delimiter=',')

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        self.phase_bias = np.genfromtxt(nnmg.phase_bias_file, delimiter=',')

        # Change body phase lag for Ex9b
        if self.phase_lag is not None:
            for i in range(self.n_oscillators_body):
                # Neighbour coupling
                if i != (self.n_oscillators_body / 2) - 1 and i < (self.n_oscillators_body - 1):
                    self.phase_bias[i][i + 1] = -self.phase_lag
                    self.phase_bias[i + 1][i] = self.phase_lag
            # All other coupling remain identical

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates = np.full(self.n_oscillators, parameters.rates)

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""

        # Unsaturated body
        if parameters.body_saturation[0] <= parameters.mlr_drive <= parameters.body_saturation[1]:
            self.nominal_amplitudes[:self.n_oscillators_body] = parameters.c_R_body[0] + \
                                                           parameters.mlr_drive * parameters.c_R_body[1]

        # Saturated body
        else:
            self.nominal_amplitudes[:self.n_oscillators_body] = parameters.R_sat

        # Unsaturated limbs
        if parameters.limb_saturation[0] <= parameters.mlr_drive <= parameters.limb_saturation[1]:
            self.nominal_amplitudes[self.n_oscillators_body:] = parameters.c_R_limb[0] + \
                                                           parameters.mlr_drive * parameters.c_R_limb[1]

        # Saturated limbs
        else:
            self.nominal_amplitudes[self.n_oscillators_body:] = parameters.R_sat

    def set_mlr_drive(self, parameters):
        self.mlr_drive = parameters.mlr_drive

