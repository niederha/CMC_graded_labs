"""Robot parameters"""

import numpy as np
import cmc_pylog as pylog
from numpy import genfromtxt
import platform

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
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        #pylog.warning("Frequencies must be set")

        #self.freqs = np.ones(self.n_oscillators)*0.01
        #self.freqs[:self.n_oscillators_body]=1
        #self.freqs[self.n_oscillators_body:]=0.3 #according to additional page 2
        self.freqs[:self.n_oscillators_body]=parameters.frequency_factor_body
        self.freqs[self.n_oscillators_body:]=parameters.frequency_factor_leg
        
    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        #pylog.warning("Coupling weights must be set")
        

        if(platform.system() == 'Linux'):
            self.coupling_weights = genfromtxt('Arrays/Coupling_Weights.csv', delimiter=',')
        else:
            self.coupling_weights = genfromtxt('Arrays\\Coupling_Weights.csv', delimiter=',')
        
    def set_phase_bias(self, parameters):
        """Set phase bias"""
        #pylog.warning("Phase bias must be set")
 
        if(platform.system() == 'Linux'):
            self.phase_bias = genfromtxt('Arrays/Phase_Shifts.csv', delimiter=',')
        else:
            self.phase_bias = genfromtxt('Arrays\\Phase_Shifts.csv', delimiter=',')
        #self.phase_bias = genfromtxt('Arrays\\Phase_Shifts.csv', delimiter=';')  
        

#        for x in parameters.phase_bias:
#            print(*x, sep="   ")
#        print(parameters.phase_bias.shape)

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        #pylog.warning("Convergence rates must be set")
        self.rates = np.ones(self.n_oscillators)*20

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        #pylog.warning("Nominal amplitudes must be set")
        #self.nominal_amplitudes = np.ones(self.n_oscillators)*np.pi/10


        self.nominal_amplitudes[:int(self.n_body_joints)] =  np.linspace(parameters.amplitude_factor_body*parameters.amplitude_factor_head_tail+parameters.steering, parameters.amplitude_factor_body, num=10)
        self.nominal_amplitudes[int(self.n_body_joints):int(self.n_oscillators_body)] =  np.linspace(parameters.amplitude_factor_body*parameters.amplitude_factor_head_tail-parameters.steering, parameters.amplitude_factor_body, num=10)

