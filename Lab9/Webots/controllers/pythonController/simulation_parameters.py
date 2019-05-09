"""Simulation parameters"""


class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.simulation_duration = 30
        self.phase_lag = None
        #self.amplitude_gradient = None

        self.amplitude_factor_head_tail = 0.5
        # Feel free to add more parameters (ex: MLR drive)
        self.drive_mlr = 2
        self.steering = 0.05 #positive -> steering to the right, negative -> steering to the left
        
        self.cv_body=[0.2,0.3]
        self.cv_leg=[0.2,0]
        self.cR_body=[0.065,0.196]
        self.cR_leg=[0.131,0.131]
        self.R_sat = 0
        self.V_sat = 0
        
        
        if (self.drive_mlr < 1 or self.drive_mlr > 5):
            self.amplitude_factor_leg = self.R_sat
            self.amplitude_factor_body = self.R_sat
            self.frequency_factor_leg = self.V_sat
            self.frequency_factor_body = self.V_sat
            
        elif (self.drive_mlr <3):
            self.amplitude_factor_body = self.cR_body[0]*self.drive_mlr+self.cR_body[1]
            self.amplitude_factor_leg = self.cR_leg[0]*self.drive_mlr+self.cR_leg[1]
            self.frequency_factor_body = self.cv_body[0]*self.drive_mlr+self.cv_body[1]
            self.frequency_factor_leg = self.cv_leg[0]*self.drive_mlr+self.cv_leg[1]
            
        else:
            self.amplitude_factor_body = self.cR_body[0]*self.drive_mlr+self.cR_body[1]
            self.frequency_factor_body = self.cv_body[0]*self.drive_mlr+self.cv_body[1]
            self.amplitude_factor_leg = self.R_sat
            self.frequency_factor_leg = self.V_sat
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations
