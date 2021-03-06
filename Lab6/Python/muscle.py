"""Implementation of Muscle class."""
from collections import namedtuple

import numpy as np


class Muscle(object):
    """This class implements the muscle model.
    The muscle model is based on the hill-type muscle model.
    """
    # Default Muscle Parameters
    C = np.log(0.05)  # pylint: disable=no-member
    N = 1.5
    K = 5.0
    TAU_ACT = 0.01  # Time constant for the activation function
    E_REF = 0.04  # Reference strain
    W = 0.4  # Shaping factor

    #: Container for results
    Result = namedtuple('Result', "time activation l_ce l_mtc v_ce "
                        "active_force passive_force tendon_force")

    def __init__(self, parameters):
        """This function initializes the muscle model.
        A default muscle name is given as muscle

        Parameters
        ----------
        parameters : <MuscleParameters>
            Instance of MuscleParameters class

        Returns:
        -------
        Muscle : <Muscle>
            Returns an instance of class Muscle

        Attributes:
        ----------
        l_MTC : float
            Length of Muscle Tendon Complex
        l_slack : float
            Tendon slack length
        l_opt : float
            Optimal fiber length
        l_CE : float
            Length of contracticle element
        v_CE : float
            Velocity of contractile element
        activeForce : float
            Active force generated by the muscle
        passiveForce : float
            Passive force generated by the muscle
        force : float
            Sum of Active and Passive forces
        tendonForce : float
            Force generated by the muscle tendon
        stim : float
            Muscle stimulation.

        Methods:
        --------
        step : func
            Integrates muscle state by time step dt

        Example:
        --------
        >>> from SystemParameters import MuscleParameters
        >>> import Muscle
        >>> muscle_parameters = MuscleParameters()
        >>> muscle1 = Muscle.Muscle(muscle_parameters)
        >>> muscle1.stim = 0.01
        >>> muscle1.step(dt)
        """

        self.parameters = parameters

        # Internal variables
        self._l_ce = 0.0
        self._v_ce = 0.0

        # Muscle specific parameters initialization
        self.L_SLACK = parameters.l_slack
        self.L_OPT = parameters.l_opt
        self.V_MAX = parameters.v_max
        self.F_MAX = parameters.f_max
        self.PENNATION = parameters.pennation

        # Muscle parameters initialization
        self._l_se = 0.0  # Muscle Series Element Length
        self._l_ce = 0.0  # Muscle Contracticle Element Length
        self._l_mtc = 0.0  # Muscle Tendon Unit (MTU) length
        self._activation = 0.05  # Muscle activation
        self.stimulation = 0.05  # base stimulation

        self.results = Muscle.Result(time=[], activation=[], l_ce=[],
                                     l_mtc=[], v_ce=[], active_force=[],
                                     passive_force=[], tendon_force=[])

    #########################  Attributes #########################

    @property
    def l_mtc(self):
        """ Length of Muscle Tendon Complex."""
        return self._l_mtc

    @l_mtc.setter
    def l_mtc(self, value):
        """ Set the length of muscle tendon complex. """
        self._l_mtc = value

    @property
    def l_ce(self):
        """ Length of muscle contracticle element."""
        return self._l_ce

    @property
    def l_se(self):
        """ Length of muscle series element."""
        return self._l_se

    @property
    def v_ce(self):
        """Velocity of muscle contracticle element"""
        return self._v_ce

    ################### METHODS FOR COMPUTING FORCES #################

    def _f_se(self, l_se):
        """This function computes the Force in the Series Element (SE).
        The function requires SE length l_SE as inputs."""
        _num = (l_se - self.L_SLACK)
        _den = self.L_SLACK * Muscle.E_REF
        f_se = ((_num/_den)**2) * (
            l_se > self.L_SLACK)
        return f_se

    def _f_pe_star(self, l_ce):
        """ This function computes the Force in the Parallel Element (PE).
        Force prevents the muscle from over-exentsion
        The function requires contracticle length l_ce as inputs."""
        if l_ce > self.L_OPT:
            return (
                (l_ce - self.L_OPT) / (self.L_OPT * Muscle.W))**2
        return 0.0

    def _f_be(self, l_ce):
        """ This function computes the Force in the muscle belly.
        Force prevents the muscle from collapsing on itself.
        The function requires SE length l_SE as inputs."""
        if l_ce <= self.L_OPT * (1.0 - Muscle.W):
            _num = (l_ce - self.L_OPT * (1.0 - Muscle.W))
            _den = (self.L_OPT * Muscle.W / 2.0)
            return (_num/_den)**2
        return 0.0

    def _f_l(self, l_ce):
        """ This function computes the force from force-length relationship.
        The function requires SE length l_SE as inputs."""
        val = abs((l_ce - self.L_OPT) / (self.L_OPT * Muscle.W))
        exposant = Muscle.C * val * val * val
        return np.exp(exposant)

    def _f_v_ce(self, v_ce):
        """ This function computes the force from force-velocity relationship.
        The function requires contracticle velocity as inputs."""
        _v_max = self.V_MAX*self.L_OPT
        if v_ce < 0.:
            return (_v_max - v_ce) / (
                _v_max + Muscle.K * v_ce)
        _num = (Muscle.N - 1) * (_v_max + v_ce)
        _den = (7.56 * Muscle.K * v_ce - _v_max)
        return Muscle.N + (_num/_den)

    def _f_v(self, f_se, f_be, act, f_l, f_pe_star):
        """ This function computes the force from force-velocity relationship.
        The function requires
        f_se : Series element force
        f_be : Muscle belly force
        act : muscle activation
        f_l : Force from force-length relationship
        f_pe_star : Parallel element force."""
        if act * f_l + f_pe_star == 0.0:
            f_v = 0.0
        else:
            f_v = (f_se + f_be) / ((act * f_l) + f_pe_star)

        f_v = 1.5 if f_v > 1.5 else f_v
        f_v = 0.0 if f_v < 0.0 else f_v

        return f_v

    def _v_ce_inv(self, f_v):
        """ This function computes the Contracticle element velocity.
        The function requires force from force-velocity relationship."""
        _v_max = self.V_MAX*self.L_OPT
        if f_v < 1.0:
            return _v_max * (1.0 - f_v) / (1.0 + f_v * Muscle.K)

        return _v_max * (f_v - 1.0) / (
            7.56 * Muscle.K * (f_v - Muscle.N) + 1.0 - Muscle.N)

    def activation_dadt(self, stimulation, activation):
        """This function updates the activation function of the muscle.
        The function requires time step dt as the inputs"""
        #: Check the bounds of the stimulation
        stimulation = max(0.05, min(1.0, stimulation))
        da_dt = (stimulation - activation) / Muscle.TAU_ACT
        return da_dt

    def initialize_muscle_length(self, l_mtc):
        """This function initializes the muscle lengths."""
        l_ce = None
        if l_mtc < (self.L_SLACK + self.L_OPT):
            l_ce = self.L_OPT
            l_se = l_mtc - self.l_ce
        else:
            if (self.L_OPT * Muscle.W + Muscle.E_REF * self.L_SLACK) != 0.0:
                _num = self.L_SLACK * (
                    self.L_OPT*Muscle.W + Muscle.E_REF*(l_mtc-self.L_OPT))
                _den = (self.L_OPT*Muscle.W + Muscle.E_REF*self.L_SLACK)
                l_se = (_num/_den)
            else:
                l_se = self.L_SLACK
            l_ce = l_mtc - l_se
        print('l_mtc : {}'.format(l_mtc))
        return l_ce

    def dxdt(self, state, time, *args):
        """ Function returns the derivative of muscle activation and
        length of the contracticle element

        Parameters:
        -----------
            - state : Muscle states
                - state[0] : <float>
                  A : Muscle activation
                - state[1] : <float>
                  l_CE : length of contracticle element

        Returns:
        --------
            - dA : Dynamics of the muscle activation
            - velocity : Muscle state derivative (
                         velocity of contracticle element)
        """
        self.stimulation = args[0]
        self.l_mtc = args[1]

        activation = state[0]
        l_ce = state[1] if state[1] > 0.0 else 0.0
        l_se = self.l_mtc - l_ce

        v_ce = self._v_ce_inv(
            self._f_v(
                self._f_se(l_se),
                self._f_be(l_ce),
                activation,
                self._f_l(l_ce),
                self._f_pe_star(l_ce)
            )
        )

        return np.array([[self.activation_dadt(self.stimulation, activation)],
                         [v_ce]])[:, 0]

    ######################### METHODS FOR LOGGING #########################

    def compute_active_force(self, l_ce, v_ce, act):
        """This function computes the Active Muscle Force.
        The function requires
        l_ce : Contracticle element length
        v_ce : Contracticle element velocity
        a : muscle activation."""
        return act * self._f_v_ce(v_ce) * self._f_l(l_ce) * self.F_MAX

    def compute_passive_force(self, l_ce):
        """ This computes the passive force generated by the muscle."""
        return (self._f_pe_star(l_ce) + self._f_be(l_ce))*self.F_MAX

    def compute_tendon_force(self, l_se):
        """This function computes the muscle tendon force.
        The function requires contracticle element length"""
        return (self._f_se(l_se))*self.F_MAX

    def instantiate_result_from_state(self, time):
        """ Instatiate the result container. """
        self.results = Muscle.Result(time=np.zeros(np.shape(time)),
                                     activation=np.zeros(np.shape(time)),
                                     l_ce=np.zeros(np.shape(time)),
                                     l_mtc=np.zeros(np.shape(time)),
                                     v_ce=np.zeros(np.shape(time)),
                                     active_force=np.zeros(np.shape(time)),
                                     passive_force=np.zeros(np.shape(time)),
                                     tendon_force=np.zeros(np.shape(time)))

    def generate_result_from_state(self, idx, time, l_mtc, state):
        """ Generate the results from the muscle states. """
        #: Clear up old results

        l_ce = state[1]
        l_se = l_mtc - l_ce
        activation = state[0]

        self.results.time[idx] = time
        self.results.l_mtc[idx] = l_mtc
        self.results.activation[idx] = activation
        self.results.l_ce[idx] = l_ce
        v_ce = self._v_ce_inv(
            self._f_v(
                self._f_se(l_se),
                self._f_be(l_ce),
                activation,
                self._f_l(l_ce),
                self._f_pe_star(l_ce)))
        self.results.v_ce[idx] = v_ce
        self.results.active_force[idx] = self.compute_active_force(l_ce,
                                                                   v_ce,
                                                                   activation)
        self.results.passive_force[idx] = self.compute_passive_force(l_ce)
        self.results.tendon_force[idx] = self.compute_tendon_force(l_se)

