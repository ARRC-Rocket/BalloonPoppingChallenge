"""Workshop-friendly baseline agent.

This module intentionally starts with the smallest useful controller for
scenario 0.  The balloons in that scenario are stationary and arranged on the
vertical launch axis, so a vertical, full-throttle flight is a meaningful
baseline before adding navigation, guidance, and closed-loop control.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


class VerticalBaselineAgent(BaseAgent):
    """Launch vertically and keep the rocket controls neutral."""

    def __init__(self, given_parameters, launch_time=1.0, throttle=1.0):
        super().__init__(given_parameters)
        self.launch_time = float(launch_time)

        throttle_min, throttle_max = self.given_parameters["rocket"]["control"][
            "throttle_range"
        ]
        self.throttle = float(np.clip(throttle, throttle_min, throttle_max))

    def get_action(self, observation):
        """Convert the current observation into one environment action."""
        simulation_time = float(observation["simulation_time"])

        return {
            "launch": simulation_time >= self.launch_time,
            # Inclination is measured from horizontal; 90 degrees is vertical.
            "launch_inclination_heading": np.array([90.0, 0.0]),
            # Zero TVC keeps thrust aligned with the rocket's body axis.
            "tvc": np.array([0.0, 0.0]),
            "roll": 0.0,
            "throttle": self.throttle,
        }
