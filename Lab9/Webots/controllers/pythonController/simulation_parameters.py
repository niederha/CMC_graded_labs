"""Simulation parameters"""

import math


class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    # Default variables taken from the paper
    default_weak_coupling = 10
    default_strong_coupling = 30
    default_limb_saturation_drives = [1., 3.]
    default_body_saturation_drives = [1., 5.]
    default_rates = 20

    def __init__(self, n_body_joints=10, n_legs_joints=4, simulation_duration=30, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = n_body_joints
        self.n_legs_joints = n_legs_joints
        self.simulation_duration = simulation_duration
        self.phase_lag = None
        self.amplitude_gradient = None
        self.weak_coupling = kwargs.pop('weak_coupling', self.default_weak_coupling)
        self.strong_coupling = kwargs.pop('strong_coupling', self.default_strong_coupling)
        self.rates = kwargs.pop('rates', self.default_rates)
        self.limb_saturation = kwargs.pop("limb_sat", self.default_limb_saturation_drives)
        self.body_saturation = kwargs.pop("body_sat", self.default_body_saturation_drives)
        self.mlr_drive = kwargs.pop('mlr_drive', 1.)
        self.phase_lag = kwargs.pop('phase_lag', None)
        self.v_sat = 0.
        self.R_sat = 0.

        self.c_freq_body = [0.2, 0.3]  # Gave up on implementing default parameters. Useless in this scope anyway
        self.c_freq_limb = [0.2, 0.0]
        # self.c_R_body = [0.01, 0.1]
        self.c_R_body = [0.065, 0.196]
        self.c_R_limb = [0.131, 0.131]

        self.update(kwargs)  # NOTE: This overrides the previous declarations

