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
        if parameters.frequency is not None:
            self.freqs[:self.n_oscillators_body] = parameters.frequency
        else:
            if (parameters.drive <= parameters.dhighBody and parameters.drive >= parameters.dlowBody):
                self.freqs[:self.n_oscillators_body]= parameters.cf1Body*parameters.drive+parameters.cf0Body
            else:
                self.freqs[:self.n_oscillators_body]= parameters.fsatBody
            
            if (parameters.drive <= parameters.dhighLimb and parameters.drive >= parameters.dlowLimb):
                self.freqs[self.n_oscillators_body:]= parameters.cf1Limb*parameters.drive+parameters.cf0Limb
            else:
                self.freqs[self.n_oscillators_body:]= parameters.fsatLimb
            
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
        
        # Change body phase lag for Ex9b
        if parameters.phase_lag is not None:
            for i in range(self.n_oscillators_body):
                # Neighbour coupling
                if i != (self.n_oscillators_body / 2) - 1 and i < (self.n_oscillators_body - 1):
                    self.phase_bias[i][i + 1] = -parameters.phase_lag
                    self.phase_bias[i + 1][i] = parameters.phase_lag
            # All other coupling remain identical
        if parameters.phase_lag_body_limb is not None:
            leg_to_body_coupling = [[1, 2, 3, 4,5],
                        [11, 12, 13, 14,15],
                        [6, 7, 8, 9],
                        [16, 17, 18, 19]]
            
            for i in range(self.n_oscillators_legs):
                for j in leg_to_body_coupling[i][:]:
                    self.phase_bias[j][self.n_oscillators_body + i] = parameters.phase_lag_body_limb
                    
        if parameters.Backwards:
            self.phase_bias *= -1 
        

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        #pylog.warning("Convergence rates must be set")
        self.rates = np.ones(self.n_oscillators)*20

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        #pylog.warning("Nominal amplitudes must be set")
        if parameters.amplitudes is not None:
            self.nominal_amplitudes[:self.n_oscillators_body] = parameters.amplitudes
            #print(self.nominal_amplitudes)
        elif parameters.amplitudesLimb is not None:
            self.nominal_amplitudes[:self.n_oscillators_body] = parameters.amplitudesLimb
            #print(self.nominal_amplitudes)
            
            if (parameters.drive <= parameters.dhighLimb and parameters.drive >= parameters.dlowLimb):
                self.nominal_amplitudes[self.n_oscillators_body:]= parameters.cR1Limb*parameters.drive+parameters.cR0Limb
            else:
                self.nominal_amplitudes[self.n_oscillators_body:]= parameters.RsatLimb
            
        else:
            if (parameters.drive <= parameters.dhighBody and parameters.drive >= parameters.dlowBody):
                self.nominal_amplitudes[:self.n_oscillators_body]= parameters.cR1Body*parameters.drive+parameters.cR0Body
            else:
                self.nominal_amplitudes[:self.n_oscillators_body]= parameters.RsatBody
            
            if (parameters.drive <= parameters.dhighLimb and parameters.drive >= parameters.dlowLimb):
                self.nominal_amplitudes[self.n_oscillators_body:]= parameters.cR1Limb*parameters.drive+parameters.cR0Limb
            else:
                self.nominal_amplitudes[self.n_oscillators_body:]= parameters.RsatLimb
            
        gradient = np.ones(self.n_oscillators)   

        gradient[:self.n_body_joints] = np.linspace(parameters.RHead,parameters.RTail,self.n_body_joints)*parameters.turnRate[0]
        gradient[self.n_body_joints:self.n_oscillators_body] = np.linspace(parameters.RHead,parameters.RTail,self.n_body_joints)*parameters.turnRate[1]      
        self.nominal_amplitudes *= gradient 
        

