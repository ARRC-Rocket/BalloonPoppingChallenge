"""
This example shows how to define agents so that they can be systematically run
in a specific environment (important for agent evaluation purpuses)

"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


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
            "launch_inclination_heading": np.array([90, 0]),
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
            "launch_inclination_heading": np.array([90, 0]),
            "tvc": np.array([torque_cmd[0], torque_cmd[1]]),
            "roll": torque_cmd[2],
            "throttle": np.array([1]),
        }


class NavigationAgent(BaseAgent):
    """A simple inertial navigation agent that integrates rocket states"""

    from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        # get launch time and target inclination/heading from kwargs
        self.launch_time = kwargs.get("launch_time", 1.0)
        self.inclination_heading = np.array(
            [kwargs.get("inclination", 90.0), kwargs.get("heading", 0.0)]
        )

        # get sensor frequency from given parameters
        self.sensor_frequency = self.given_parameters["rocket"]["sensors"][
            "sampling_rate"
        ]

        # Calculate initial attitude quaternion from desired inclination and heading
        e0, e1, e2, e3 = get_initial_attitude(
            self.inclination_heading[0], self.inclination_heading[1]
        )
        # State vector: [x, y, z, vx, vy, vz, e0, e1, e2, e3]
        self.states = np.array([
            0,  # x position
            0,  # y position
            self.given_parameters["environment"]["elevation"],  # z position
            0,  # x velocity
            0,  # y velocity
            0,  # z velocity
            e0,  # quaternion e0
            e1,  # quaternion e1
            e2,  # quaternion e2
            e3,  # quaternion e3
        ])

        self.gyro_prev = np.array([0.0, 0.0, 0.0])
        self.vel_prev = np.array([0.0, 0.0, 0.0])
        self.accel_prev = np.array([0.0, 0.0, 0.0])
        self.gravity = np.array([0.0, 0.0, -9.81])  # gravity vector in local launch frame (constant model)

    def get_action(self, observation):
        """compute agent's action given observation

        This function is necessary to define as it overrides
        an abstract method
        """
        if observation["simulation_time"] >= self.launch_time:
            launch = True
        else:
            launch = False

        if not np.isnan(observation["rocket_sensors"][:3]).any():
            # start navigation after launch
            gyro = observation["rocket_sensors"][:3] - self.given_parameters["rocket"]["sensors"]["gyro_constant_bias"]
            accel = observation["rocket_sensors"][3:6] - self.given_parameters["rocket"]["sensors"]["accelerometer_constant_bias"]

            # Step 1: Update attitude
            gyro_increment = (gyro + self.gyro_prev) * (1 / self.sensor_frequency) / 2
            angle = np.linalg.norm(gyro_increment)
            dw = np.cos(angle / 2)
            dx, dy, dz = np.sin(angle / 2) * gyro_increment / angle

            e0, e1, e2, e3 = self.states[6:10]
            # Hamilton right-multiplication: q_new = q_old ⊗ q_delta
            new_e0 = e0 * dw - e1 * dx - e2 * dy - e3 * dz
            new_e1 = e0 * dx + e1 * dw + e2 * dz - e3 * dy
            new_e2 = e0 * dy - e1 * dz + e2 * dw + e3 * dx
            new_e3 = e0 * dz + e1 * dy - e2 * dx + e3 * dw

            norm = np.sqrt(new_e0**2 + new_e1**2 + new_e2**2 + new_e3**2)
            self.states[6:10] = np.array([new_e0 / norm, new_e1 / norm, new_e2 / norm, new_e3 / norm])

            # Step 2: Update velocity and position
            # Rotate body-frame accel to local frame via q ⊗ v_pure ⊗ q* (Rodrigues form)
            q_vec = np.array(self.states[7:10])  # [e1, e2, e3]
            t = 2.0 * np.cross(q_vec, accel)
            accel_lframe = accel + self.states[6] * t + np.cross(q_vec, t) - self.gravity
            accel_increment = (accel_lframe + self.accel_prev) * (1 / self.sensor_frequency) / 2
            self.states[3:6] = np.array([self.states[i + 3] + accel_increment[i] for i in range(3)])
            vel_increment = (self.states[3:6] + self.vel_prev) * (1 / self.sensor_frequency) / 2
            self.states[0:3] = np.array([self.states[i] + vel_increment[i] for i in range(3)])

            self.gyro_prev = gyro
            self.vel_prev = self.states[3:6]
            self.accel_prev = accel_lframe

        return {
            "launch": launch,
            "launch_inclination_heading": self.inclination_heading,
            "tvc": np.array([0.0, 0.0]),
            "roll": 0.0,
            "throttle": np.array([1]),
        }
