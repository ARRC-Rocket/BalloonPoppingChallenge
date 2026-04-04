"""
This example shows how to define agents so that they can be systematically run
in a specific environment (important for agent evaluation purpuses)

"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


class RollControlAgent(BaseAgent):
    """An agent that controls roll using a PID controller and launches at t=1s"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # Initialize an array to store roll rate errors for PID control
        self.roll_rate_errors = np.array([])
        # Default target roll rate is 0 rad/s
        self.roll_rate_target = kwargs.get("roll_rate_target", 0.0)

    def get_action(self, observation):
        """compute agent's action given observation

        This function is necessary to define as it overrides
        an abstract method
        """
        sensor_frequency = self.given_parameters["rocket"]["sensors"]["sampling_rate"]

        if observation["simulation_time"] >= 1.0:
            launch = True
        else:
            launch = False

        if not np.isnan(observation["rocket_sensors"][:3]).any():
            KP = 50.0
            KI = 1.0
            KD = 0.0

            self.roll_rate_errors = np.append(
                self.roll_rate_errors,
                self.roll_rate_target - observation["rocket_sensors"][2],
            )
            roll_rate_error_integral = np.sum(self.roll_rate_errors) / sensor_frequency
            roll_rate_error_derivative = (
                (self.roll_rate_errors[-1] - self.roll_rate_errors[-2])
                * sensor_frequency
                if len(self.roll_rate_errors) > 1
                else 0
            )
            roll_torque_cmd = (
                KP * self.roll_rate_errors[-1]
                + KI * roll_rate_error_integral
                + KD * roll_rate_error_derivative
            )
        else:
            roll_torque_cmd = 0.0

        return {
            "launch": launch,
            "launch_inclination_heading": np.array([90, 180]),
            "tvc": np.array([0, 0]),
            "roll": roll_torque_cmd,
            "throttle": np.array([1]),
        }


class AttitudeRateControlAgent(BaseAgent):
    """An agent that controls attitude rates using a PID controller and launches at t=1s"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # Initialize an array to store attitude rate errors for PID control angular rates of (x, y, z) axis
        # Shape: (3, n_timesteps) where each row tracks one axis over time
        self.rate_errors = np.zeros((3, 1))
        # Default target are 0 rad/s
        self.rate_targets = kwargs.get("rate_targets", [0.0, 0.0, 0.0])
        self.launch_time = kwargs.get("launch_time", 1.0)

    def get_action(self, observation):
        """compute agent's action given observation

        This function is necessary to define as it overrides
        an abstract method
        """
        sensor_frequency = self.given_parameters["rocket"]["sensors"]["sampling_rate"]

        if observation["simulation_time"] >= self.launch_time:
            launch = True
        else:
            launch = False

        if not np.isnan(observation["rocket_sensors"][:3]).any():
            KP = [100.0, 100.0, 100.0]
            KI = [0.0, 0.0, 5.0]
            KD = [0.0, 0.0, 0.0]

            self.rate_errors = np.append(
                self.rate_errors,
                (self.rate_targets - observation["rocket_sensors"][:3]).reshape(-1, 1),
                axis=1,
            )
            roll_rate_error_integrals = (
                np.sum(self.rate_errors, axis=1) / sensor_frequency
            )
            roll_rate_error_derivatives = (
                (self.rate_errors[:, -1] - self.rate_errors[:, -2]) * sensor_frequency
                if self.rate_errors.shape[1] > 1
                else np.array([0.0, 0.0, 0.0])
            )
            torque_cmd = (
                KP * self.rate_errors[:, -1]
                + KI * roll_rate_error_integrals[:3]
                + KD * roll_rate_error_derivatives[:3]
            )
        else:
            torque_cmd = np.array([0.0, 0.0, 0.0])

        return {
            "launch": launch,
            "launch_inclination_heading": np.array([90, 180]),
            "tvc": np.array([torque_cmd[0], torque_cmd[1]]),
            "roll": torque_cmd[2],
            "throttle": np.array([1]),
        }
