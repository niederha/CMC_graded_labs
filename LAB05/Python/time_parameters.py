
class TimeParameters:
    """Used to pass the time infos for the simulation in an easier way"""
    def __init__(self, time_start=0.0, time_stop=0.2, time_step=0.001, time_stabilize=0.2):
        self.t_start = None
        self.t_stop = None
        self.t_step = None
        self.t_start = time_start
        self.t_stop = time_stop
        self.t_step = time_step
        self.times = self.recompute_times()
        self.t_stabilize = time_stabilize

    def recompute_times(self):
        if self.t_start is not None and self.t_stop is not None and self.t_step is not None:
            return np.arange(self.t_start, self.t_stop, self.t_step)

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, new_t_start):
        self._t_start = new_t_start
        if new_t_start is not None:
            self.times = self.recompute_times()

    @property
    def t_stop(self):
        return self._t_stop

    @t_stop.setter
    def t_stop(self, new_t_stop):
        self._t_stop = new_t_stop
        if new_t_stop is not None:
            self.times = self.recompute_times()

    @property
    def t_step(self):
        return self._t_step

    @t_step.setter
    def t_step(self, new_t_step):
        self._t_step = new_t_step
        if new_t_step is not None:
            self.times = self.recompute_times()
