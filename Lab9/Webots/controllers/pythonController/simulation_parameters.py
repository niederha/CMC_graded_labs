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
        self.amplitude_gradient = None
        self.amplitudes = None
        self.frequency = None
        # Feel free to add more parameters (ex: MLR drive)
        #self.drive_mlr = 0.1
        #self.drive= 0
        
        
        
        self.dlowBody = 1
        self.dhighBody = 5
        self.dlowLimb = 1
        self.dhighLimb = 3
        
        self.cf1Limb = 0.2 
        self.cf0Limb = 0.0
        self.cR1Limb = 0.131
        self.cR0Limb = 0.131
        self.cf1Body = 0.2
        self.cf0Body = 0.3
        self.cR1Body = 0.065
        self.cR0Body = 0.196
        
        self.fsatBody = 0.0
        self.fsatLimb = 0.0
        self.RsatBody = 0.0
        self.RsatLimb =0.0
        
        self.RHead = 1
        self.RTail = 1
        
        self.Backwards = False
        
        self.turnRate = [1,1]
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

