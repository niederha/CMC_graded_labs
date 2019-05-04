"""Robot parameters"""

import numpy as np
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
        pylog.warning("Coupling weights must be set")

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        self.coupling_weights = np.zeros([self.n_oscillators, self.n_oscillators])
        
        body_weight = 10
        leg_weight = 30
        
        # Coupling weights between body joints left side
        self.coupling_weights[:self.n_body_joints,:self.n_body_joints]=(body_weight*np.eye(self.n_body_joints,k=-1)+body_weight*np.eye(self.n_body_joints,k=1))
        # Coupling weights between body joints right side
        self.coupling_weights[self.n_body_joints:self.n_oscillators_body,self.n_body_joints:self.n_oscillators_body]=(body_weight*np.eye(self.n_body_joints,k=-1)+body_weight*np.eye(self.n_body_joints,k=1))
        # Coupling weights between body joints between left and right side
        self.coupling_weights[:self.n_oscillators_body,:self.n_oscillators_body]+=body_weight*np.eye(self.n_oscillators_body,k=-self.n_body_joints)+body_weight*np.eye(self.n_oscillators_body,k=self.n_body_joints)
        
        # Coupling weights between leg joints
        self.coupling_weights[self.n_oscillators_body:self.n_oscillators,self.n_oscillators_body:self.n_oscillators]=body_weight*np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
        
        # Coupling weights from front left leg to body
        self.coupling_weights[self.n_oscillators_body,:int(self.n_body_joints/2)]=leg_weight*np.ones([1,int(self.n_body_joints/2)])
        # Coupling weights from front right leg to body
        self.coupling_weights[self.n_oscillators_body+1,self.n_body_joints:3*int(self.n_body_joints/2)]=leg_weight*np.ones([1,int(self.n_body_joints/2)])      
         # Coupling weights from back left leg to body
        self.coupling_weights[self.n_oscillators_body+2,int(self.n_body_joints/2):self.n_body_joints]=leg_weight*np.ones([1,int(self.n_body_joints/2)])          
        # Coupling weights from back left leg to body
        self.coupling_weights[self.n_oscillators_body+3,3*int(self.n_body_joints/2):2*self.n_body_joints]=leg_weight*np.ones([1,int(self.n_body_joints/2)])        
        #pylog.warning("Coupling weights must be set")

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        
        # Phase bias between body joints left side
        self.phase_bias[:self.n_body_joints,:self.n_body_joints]=(-2*np.pi/self.n_body_joints*np.eye(self.n_body_joints,k=-1)+2*np.pi/self.n_body_joints*np.eye(self.n_body_joints,k=1))
        # Phase bias between body joints right side
        self.phase_bias[self.n_body_joints:self.n_oscillators_body,self.n_body_joints:self.n_oscillators_body]=(-2*np.pi/self.n_body_joints*np.eye(self.n_body_joints,k=-1)+2*np.pi/self.n_body_joints*np.eye(self.n_body_joints,k=1))
        # Phase bias between body joints between left and right side
        self.phase_bias[:self.n_oscillators_body,:self.n_oscillators_body]+=np.pi*np.eye(self.n_oscillators_body,k=-self.n_body_joints)+np.pi*np.eye(self.n_oscillators_body,k=self.n_body_joints)
        
        # Phase bias between front leg joints
        self.phase_bias[self.n_oscillators_body:self.n_oscillators-2,self.n_oscillators_body:self.n_oscillators-2]=[[0,np.pi],[np.pi,0]]
        
        # Phase bias between back leg joints
        self.phase_bias[self.n_oscillators_body+2:self.n_oscillators,self.n_oscillators_body+2:self.n_oscillators]=[[0,np.pi],[np.pi,0]]
        #pylog.warning("Phase bias must be set")

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        pylog.warning("Convergence rates must be set")

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        pylog.warning("Nominal amplitudes must be set")

