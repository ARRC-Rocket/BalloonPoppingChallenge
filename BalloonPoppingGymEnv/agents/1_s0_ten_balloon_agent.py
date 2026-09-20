"""Closed-loop Scenario 0 agent used as the workshop's second example."""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


class TenBalloonRateControlAgent(BaseAgent):
    """Keep body rates near zero while flying through the vertical balloon line.

    Scenario 0 places all ten balloons on the vertical launch axis.  Unlike the
    open-loop baseline, this agent uses gyroscope feedback to reject rotations
    that would make the rocket drift away from that axis.
    """

    def __init__(
        self,
        given_parameters,
        launch_time=1.0,
        pitch_yaw_kp=100.0,
        roll_kp=100.0,
        roll_ki=5.0,
    ):
        super().__init__(given_parameters)
        self.launch_time = float(launch_time)
        self.pitch_yaw_kp = float(pitch_yaw_kp)
        self.roll_kp = float(roll_kp)
        self.roll_ki = float(roll_ki)

        control = self.given_parameters["rocket"]["control"]
        self.gimbal_limit = float(control["max_gimbal_angle"])
        self.roll_torque_limit = float(control["max_roll_torque"])
        self.dt = float(self.given_parameters["simulation"]["time_step"])
        self.gimbal_step = 0.999 * float(control["gimbal_rate_limit"]) * self.dt
        self.roll_step = 0.999 * float(control["torque_rate_limit"]) * self.dt
        self.roll_error_integral = 0.0
        self.last_tvc = np.zeros(2)
        self.last_roll = 0.0

    def get_action(self, observation):
        simulation_time = float(observation["simulation_time"])
        gyro = np.asarray(observation["rocket_sensors"][:3], dtype=float)

        tvc_command = np.zeros(2)
        roll_command = 0.0

        # Sensor values are NaN before launch, so feedback starts only after the
        # first valid gyroscope sample arrives.
        if np.all(np.isfinite(gyro)):
            rate_error = -gyro

            tvc_command = np.clip(
                self.pitch_yaw_kp * rate_error[:2],
                -self.gimbal_limit,
                self.gimbal_limit,
            )
            tvc_command = np.clip(
                tvc_command,
                self.last_tvc - self.gimbal_step,
                self.last_tvc + self.gimbal_step,
            )

            proposed_integral = self.roll_error_integral + rate_error[2] * self.dt
            unsaturated_roll = (
                self.roll_kp * rate_error[2] + self.roll_ki * proposed_integral
            )
            roll_command = float(
                np.clip(
                    unsaturated_roll,
                    -self.roll_torque_limit,
                    self.roll_torque_limit,
                )
            )
            roll_command = float(
                np.clip(
                    roll_command,
                    self.last_roll - self.roll_step,
                    self.last_roll + self.roll_step,
                )
            )

            # Conditional integration prevents the integral term from winding
            # up while the actuator is saturated.
            if abs(unsaturated_roll) <= self.roll_torque_limit:
                self.roll_error_integral = proposed_integral

            self.last_tvc = tvc_command
            self.last_roll = roll_command

        return {
            "launch": simulation_time >= self.launch_time,
            "launch_inclination_heading": np.array([90.0, 0.0]),
            "tvc": tvc_command,
            "roll": roll_command,
            "throttle": 1.0,
        }
