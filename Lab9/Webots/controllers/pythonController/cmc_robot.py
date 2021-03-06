"""CMC robot"""

import numpy as np
from network import SalamanderNetwork
from experiment_logger import ExperimentLogger

from controller import Keyboard

class SalamanderCMC(object):
    """Salamander robot for CMC"""

    N_BODY_JOINTS = 10
    N_LEGS = 4

    def __init__(self, robot, n_iterations, parameters, logs="logs/log.npz"):
        super(SalamanderCMC, self).__init__()
        self.robot = robot
        timestep = int(robot.getBasicTimeStep())
        self.network = SalamanderNetwork(1e-3*timestep, parameters)

        # Position sensors
        self.position_sensors = [
            self.robot.getPositionSensor('position_sensor_{}'.format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ] + [
            self.robot.getPositionSensor('position_sensor_leg_{}'.format(i+1))
            for i in range(self.N_LEGS)
        ]
        for sensor in self.position_sensors:
            sensor.enable(timestep)

        # GPS
        self.gps = robot.getGPS("fgirdle_gps")
        self.gps.enable(timestep)

        # Get motors
        self.motors_body = [
            self.robot.getMotor("motor_{}".format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ]
        self.motors_legs = [
            self.robot.getMotor("motor_leg_{}".format(i+1))
            for i in range(self.N_LEGS)
        ]

        # Set motors
        for motor in self.motors_body:
            motor.setPosition(0)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)
        for motor in self.motors_legs:
            motor.setPosition(-np.pi/2)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)

        # Iteration counter
        self.iteration = 0

        # Logging
        self.log = ExperimentLogger(
            n_iterations,
            n_links=1,
            n_joints=self.N_BODY_JOINTS+self.N_LEGS,
            filename=logs,
            timestep=1e-3*timestep,
            **parameters
        )
        
        #GPS stuff
        
        self.waterPosx = 0
        self.NetworkParameters = self.network.parameters
        self.SimulationParameters = parameters
        self.doTransition = False

        self.keyboard = Keyboard()
        self.keyboard.enable(samplingPeriod=100)
        self.lastkey = 0
        
        

    def log_iteration(self):
        """Log state"""
        self.log.log_link_positions(self.iteration, 0, self.gps.getValues())
        for i, motor in enumerate(self.motors_body):
            # Position
            self.log.log_joint_position(
                self.iteration, i,
                self.position_sensors[i].getValue()
            )
            # # Velocity
            # self.log.log_joint_velocity(
            #     self.iteration, i,
            #     motor.getVelocity()
            # )
            # Command
            self.log.log_joint_cmd(
                self.iteration, i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
        for i, motor in enumerate(self.motors_legs):
            # Position
            self.log.log_joint_position(
                self.iteration, 10+i,
                self.position_sensors[10+i].getValue()
            )
            # # Velocity
            # self.log.log_joint_velocity(
            #     self.iteration, i,
            #     motor.getVelocity()
            # )
            # Command
            self.log.log_joint_cmd(
                self.iteration, 10+i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )

    def step(self):
        """Step"""
        # Increment iteration
        self.iteration += 1

        # Update network
        self.network.step()
        positions = self.network.get_motor_position_output()

        # Update control
        for i in range(self.N_BODY_JOINTS):
            self.motors_body[i].setPosition(positions[i])
        for i in range(self.N_LEGS):
            self.motors_legs[i].setPosition(
                positions[self.N_BODY_JOINTS+i] - np.pi/2
            )
        
        
        key=self.keyboard.getKey()
        if (key==ord('A') and key is not self.lastkey):
            print('Turning left')
            self.SimulationParameters.turnRate = [0.5,1]
            self.NetworkParameters.set_nominal_amplitudes(self.SimulationParameters)
            self.lastkey = key
        if (key==ord('D') and key is not self.lastkey):
            print('Turning right')
            self.SimulationParameters.turnRate = [1,0.5]
            self.NetworkParameters.set_nominal_amplitudes(self.SimulationParameters)
            self.lastkey = key
        if (key==ord('W') and key is not self.lastkey):
            print('Going forward')
            self.SimulationParameters.turnRate = [1,1]
            self.NetworkParameters.set_nominal_amplitudes(self.SimulationParameters)
            self.SimulationParameters.Backwards = False
            self.NetworkParameters.set_phase_bias(self.SimulationParameters)
            self.lastkey = key
            
        if (key==ord('S') and key is not self.lastkey):
            print('Going backward')
            self.SimulationParameters.Backwards = True
            self.NetworkParameters.set_phase_bias(self.SimulationParameters)
            self.SimulationParameters.turnRate = [1,1]
            self.NetworkParameters.set_nominal_amplitudes(self.SimulationParameters)
            self.lastkey = key
            
        if (key==ord('T') and key is not self.lastkey):
            if self.doTransition:
                print('Disabling transition')
                self.doTransition = False
            else:
                print('Enabling transition')
                self.doTransition = True
            self.lastkey = key
        
        if self.doTransition:
            gpsPos = self.gps.getValues()
            
            if gpsPos[0] < self.waterPosx+2 and gpsPos[0] > self.waterPosx -0.5:
                gain = 4/2.5*(gpsPos[0]+0.5) + 1
                self.SimulationParameters.drive = gain
                #print('Transitioning')
                
                self.NetworkParameters.set_nominal_amplitudes(self.SimulationParameters)
                self.NetworkParameters.set_frequencies(self.SimulationParameters)

        # Log data
        self.log_iteration()

