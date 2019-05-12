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
        
        """
        travelingWave = 10
        legInPhase = 30
        legAntiPhase = 10
        
#        travelingWave = 1
#        legInPhase = 2
#        legAntiPhase = 3
        
        
        size = int(self.n_oscillators_body/2)
        #size = 6
        
        self.coupling_weights = np.zeros([self.n_oscillators,self.n_oscillators])
        #parameters.phase_bias = np.zeros([size*3,size*3])
        self.coupling_weights[:2*size,:2*size] = travelingWave*(    np.eye(2*size,k=-1) 
                                                                        + np.eye(2*size,k=1)     
                                                                        + np.eye(2*size,k=size) 
                                                                        + np.eye(2*size,k=-size))
        
        self.coupling_weights[int(size*2),:int(size/2)] = legInPhase
        self.coupling_weights[int(size*2),int(size*2)+1] = legAntiPhase
        self.coupling_weights[int(size*2),int(size*2)+2] = legAntiPhase
        
        self.coupling_weights[int(size*2)+1,size:int(3*size/2)] = legInPhase
        self.coupling_weights[int(size*2)+1,int(size*2)] = legAntiPhase
        self.coupling_weights[int(size*2)+1,int(size*2)+3] = legAntiPhase
        
        self.coupling_weights[int(size*2)+2,int(size/2):size] = legInPhase
        self.coupling_weights[int(size*2)+2,int(size*2)] = legAntiPhase
        self.coupling_weights[int(size*2)+2,int(size*2)+3] = legAntiPhase
        
        self.coupling_weights[int(size*2)+3,int(3*size/2):int(2*size)] = legInPhase
        self.coupling_weights[int(size*2)+3,int(size*2)+1] = legAntiPhase
        self.coupling_weights[int(size*2)+3,int(size*2)+2] = legAntiPhase
     
#        for x in parameters.coupling_weights:
#            print(*x, sep="   ")
#        print(parameters.phase_bias.shape)
        """
        if(platform.system() == 'Linux'):
            self.coupling_weights = genfromtxt('Arrays/Coupling_Weights.csv', delimiter=',')
        else:
            self.coupling_weights = genfromtxt('Arrays\\Coupling_Weights.csv', delimiter=';')
        
    def set_phase_bias(self, parameters):
        """Set phase bias"""
        #pylog.warning("Phase bias must be set")
        """
        travelingWave = 2*np.pi/self.n_joints
        legInPhase = 0
        legAntiPhase = np.pi
        
#        travelingWave = 1
#        legInPhase = 2
#        legAntiPhase = 3
        
        
        size = int(self.n_oscillators_body/2)
        #size = 6
        
        self.phase_bias = np.zeros([self.n_oscillators,self.n_oscillators])
        #parameters.phase_bias = np.zeros([size*3,size*3])
        self.phase_bias[:2*size,:2*size] = ( - travelingWave*np.eye(2*size,k=-1) 
                                + travelingWave*np.eye(2*size,k=1)        
                                + legAntiPhase* np.eye(2*size,k=size)   
                                + legAntiPhase* np.eye(2*size,k=-size))
        
        self.phase_bias[int(size*2),:int(size/2)] = legInPhase
        self.phase_bias[int(size*2),int(size*2)+1] = legAntiPhase
        self.phase_bias[int(size*2),int(size*2)+2] = legAntiPhase
        
        self.phase_bias[int(size*2)+1,size:int(3*size/2)] = legInPhase
        self.phase_bias[int(size*2)+1,int(size*2)] = -legAntiPhase*(-1)
        self.phase_bias[int(size*2)+1,int(size*2)+3] = -legAntiPhase*(-1)
        
        self.phase_bias[int(size*2)+2,int(size/2):size] = legInPhase
        self.phase_bias[int(size*2)+2,int(size*2)] = -legAntiPhase*(-1)
        self.phase_bias[int(size*2)+2,int(size*2)+3] = -legAntiPhase*(-1)
        
        self.phase_bias[int(size*2)+3,int(3*size/2):int(2*size)] = legInPhase
        self.phase_bias[int(size*2)+3,int(size*2)+1] = legAntiPhase
        self.phase_bias[int(size*2)+3,int(size*2)+2] = legAntiPhase

    
        """     
        if(platform.system() == 'Linux'):
            self.phase_bias = genfromtxt('Arrays/Phase_Shifts.csv', delimiter=',')
        else:
            self.phase_bias = genfromtxt('Arrays\\Phase_Shifts.csv', delimiter=';')
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
        #self.nominal_amplitudes = np.ones(self.n_oscillators)*np.pi/8
        
        if (parameters.drive <= parameters.dhighBody and parameters.drive >= parameters.dlowBody):
            self.nominal_amplitudes[:self.n_oscillators_body]= parameters.cR1Body*parameters.drive+parameters.cR0Body
        else:
            self.nominal_amplitudes[:self.n_oscillators_body]= parameters.RsatBody
        
        if (parameters.drive <= parameters.dhighLimb and parameters.drive >= parameters.dlowLimb):
            self.nominal_amplitudes[self.n_oscillators_body:]= parameters.cR1Limb*parameters.drive+parameters.cR0Limb
        else:
            self.nominal_amplitudes[self.n_oscillators_body:]= parameters.RsatLimb
        
