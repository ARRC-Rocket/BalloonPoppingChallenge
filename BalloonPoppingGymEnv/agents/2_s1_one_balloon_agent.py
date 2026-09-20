"""A small Scenario 1 agent that commits to one moving balloon.

The competition agent plans routes through several balloons.  This workshop
version keeps only the first learning loop: observe one target, predict where
it will be, and steer the rocket toward that intercept point.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


GRAVITY = np.array([0.0, 0.0, -9.80665])
EPSILON = 1e-9


def unit(vector):
    """Return a unit vector, or vertical if the input is too small."""
    norm = float(np.linalg.norm(vector))
    if norm < EPSILON:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(vector, dtype=float) / norm


def rotation_matrix(quaternion):
    """Convert a body-to-launch-frame quaternion to a rotation matrix."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def integrate_attitude(quaternion, gyro, dt):
    """Integrate one body-rate sample with a quaternion update."""
    rotation = np.asarray(gyro, dtype=float) * dt
    angle = float(np.linalg.norm(rotation))
    if angle < EPSILON:
        return quaternion

    half_angle = angle / 2
    dw = np.cos(half_angle)
    dx, dy, dz = np.sin(half_angle) * rotation / angle
    w, x, y, z = quaternion
    updated = np.array(
        [
            w * dw - x * dx - y * dy - z * dz,
            w * dx + x * dw + y * dz - z * dy,
            w * dy - x * dz + y * dw + z * dx,
            w * dz + x * dy - y * dx + z * dw,
        ]
    )
    return updated / np.linalg.norm(updated)


class OneBalloonAgent(BaseAgent):
    """Choose one reachable balloon at launch and pursue only that balloon."""

    def __init__(self, given_parameters, launch_time=49.5):
        super().__init__(given_parameters)
        self.launch_time = float(launch_time)
        self.dt = float(given_parameters["simulation"]["time_step"])
        self.elevation = float(given_parameters["environment"]["elevation"])

        rocket = given_parameters["rocket"]
        control = rocket["control"]
        self.max_gimbal = float(control["max_gimbal_angle"])
        self.gimbal_rate = float(control["gimbal_rate_limit"])
        self.max_roll = float(control["max_roll_torque"])
        self.roll_rate = float(control["torque_rate_limit"])

        # A conservative acceleration limit is enough for choosing a target.
        # Closed-loop throttle remains full so students only tune guidance first.
        body_mass = float(rocket["rocket_body"]["mass"])
        liquid_mass = float(rocket["tank"]["initial_liquid_mass"])
        gas_mass = float(rocket["tank"]["initial_gas_mass"])
        self.thrust_acceleration = float(rocket["motor"]["thrust_source"]) / (
            body_mass + liquid_mass + gas_mass
        )

        balloon_count = int(given_parameters["balloon"]["num"])
        self.previous_balloon_velocity = np.zeros((balloon_count, 3))
        self.balloon_acceleration = np.zeros((balloon_count, 3))
        self.previous_status = np.zeros(balloon_count, dtype=int)

        self.target = None
        self.intercept_time = None
        self.launch_angles = np.array([90.0, 0.0])
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_tvc = np.zeros(2)
        self.last_roll = 0.0

    def _observe_balloons(self, states, status):
        """Estimate target acceleration from successive velocity observations."""
        released_twice = (status == 1) & (self.previous_status == 1)
        if np.any(released_twice):
            measured = (
                states[released_twice, 3:6]
                - self.previous_balloon_velocity[released_twice]
            ) / self.dt
            measured_norm = np.linalg.norm(measured, axis=1, keepdims=True)
            measured *= np.minimum(1.0, 1.0 / np.maximum(measured_norm, EPSILON))
            self.balloon_acceleration[released_twice] = (
                0.98 * self.balloon_acceleration[released_twice] + 0.02 * measured
            )

        self.previous_balloon_velocity = states[:, 3:6].copy()
        self.previous_status = status.copy()

    def _predict_balloon(self, state, acceleration, time_to_go):
        return state[:3] + state[3:6] * time_to_go + 0.5 * acceleration * time_to_go**2

    def _choose_target(self, states, status):
        """Select the easiest single intercept under a constant-acceleration model."""
        candidates = np.flatnonzero(status == 1)
        flight_times = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
        launch_position = np.array([0.0, 0.0, self.elevation])
        best = None

        for target in candidates:
            for flight_time in flight_times:
                target_position = self._predict_balloon(
                    states[target], self.balloon_acceleration[target], flight_time
                )
                net_acceleration = (
                    2.0 * (target_position - launch_position) / flight_time**2
                )
                thrust_vector = net_acceleration - GRAVITY
                required_thrust = float(np.linalg.norm(thrust_vector))
                axis = unit(thrust_vector)
                inclination = float(np.degrees(np.arcsin(axis[2])))
                closing_speed = float(
                    np.linalg.norm(net_acceleration * flight_time - states[target, 3:6])
                )

                reachable = (
                    required_thrust <= 1.04 * self.thrust_acceleration
                    and inclination >= 65.0
                    and closing_speed <= 25.0
                )
                if not reachable:
                    continue

                # Short flights are easier to predict; spare thrust gives the
                # feedback controller room to correct wind and model error.
                score = (
                    flight_time
                    + 2.0 * required_thrust / self.thrust_acceleration
                    + 0.05 * (90.0 - inclination)
                )
                if best is None or score < best[0]:
                    best = (score, int(target), flight_time, axis)

        if best is None:
            return

        _, self.target, flight_time, axis = best
        self.intercept_time = self.launch_time + flight_time
        inclination = np.degrees(np.arcsin(np.clip(axis[2], -1.0, 1.0)))
        heading = np.degrees(np.arctan2(axis[0], axis[1])) % 360.0
        self.launch_angles = np.array([inclination, heading])
        self.quaternion = np.asarray(get_initial_attitude(*self.launch_angles))

    def _guidance(self, time, states, rocket_position, rocket_velocity):
        """Solve p_target = p + v*t + 1/2*a*t^2 for the required acceleration."""
        time_to_go = float(np.clip(self.intercept_time - time, 0.25, 7.0))
        target_state = states[self.target]
        target_position = self._predict_balloon(
            target_state, self.balloon_acceleration[self.target], time_to_go
        )
        acceleration = (
            2.0
            * (target_position - rocket_position - rocket_velocity * time_to_go)
            / time_to_go**2
        )

        # Near the target, correct the remaining zero-effort miss instead of
        # trusting the original intercept time forever.
        relative_position = target_state[:3] - rocket_position
        distance = float(np.linalg.norm(relative_position))
        if distance < 35.0:
            relative_velocity = target_state[3:6] - rocket_velocity
            closest_time = float(
                np.clip(
                    -np.dot(relative_position, relative_velocity)
                    / max(np.dot(relative_velocity, relative_velocity), EPSILON),
                    0.5,
                    1.5,
                )
            )
            miss = relative_position + relative_velocity * closest_time
            correction = 2.5 * miss / closest_time**2
            correction_norm = float(np.linalg.norm(correction))
            correction *= min(1.0, 4.0 / max(correction_norm, EPSILON))
            acceleration += np.clip((35.0 - distance) / 22.0, 0.0, 0.7) * correction

        return acceleration

    def _control(self, desired_acceleration, gyro):
        """Point the rocket's thrust axis along the requested acceleration."""
        attitude = rotation_matrix(self.quaternion)
        current_axis = attitude[:, 2]
        desired_axis = unit(desired_acceleration - GRAVITY)
        axis_error_body = attitude.T @ np.cross(current_axis, desired_axis)

        desired_rate = np.clip(2.8 * axis_error_body, -1.35, 1.35)
        tvc = np.clip(
            10.0 * (desired_rate[:2] - gyro[:2]),
            -self.max_gimbal,
            self.max_gimbal,
        )
        tvc_step = 0.999 * self.gimbal_rate * self.dt
        tvc = np.clip(tvc, self.last_tvc - tvc_step, self.last_tvc + tvc_step)

        roll = float(np.clip(-5.0 * gyro[2], -self.max_roll, self.max_roll))
        roll_step = 0.999 * self.roll_rate * self.dt
        roll = float(
            np.clip(roll, self.last_roll - roll_step, self.last_roll + roll_step)
        )
        self.last_tvc = tvc
        self.last_roll = roll
        return tvc, roll

    def get_action(self, observation):
        """Map one observation to one legal action."""
        time = float(observation["simulation_time"])
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)
        self._observe_balloons(states, status)

        launch_now = time >= self.launch_time and self.target is None
        if launch_now:
            self._choose_target(states, status)
            launch_now = self.target is not None

        tvc = np.zeros(2)
        roll = 0.0
        sensors = np.asarray(observation["rocket_sensors"], dtype=float)
        if self.target is not None and np.all(np.isfinite(sensors)):
            gyro = sensors[:3]
            self.quaternion = integrate_attitude(self.quaternion, gyro, self.dt)
            desired_acceleration = self._guidance(
                time, states, sensors[6:9], sensors[9:12]
            )
            tvc, roll = self._control(desired_acceleration, gyro)

        return {
            "launch": launch_now,
            "launch_inclination_heading": self.launch_angles.copy(),
            "tvc": tvc,
            "roll": roll,
            "throttle": 1.0,
        }
