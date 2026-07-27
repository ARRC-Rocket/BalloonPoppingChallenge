"""The coordinate datum the competition API exposes.

X and Y are measured from the launch point, but Z is altitude above sea level:
the rocket starts at ``z = elevation`` and a balloon 10 m above the pad reads
``elevation + 10``. Upstream RocketPy documents launch-site-relative positions
instead, so this is a fork-level divergence that a future submodule update could
silently reverse. Competitors aim with these numbers, so it is worth pinning.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestCoordinateDatum(unittest.TestCase):
    def setUp(self):
        self.parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        self.elevation = float(self.parameters["environment"]["elevation"])
        self.env = BalloonPoppingEnv(render_mode=None, parameters=self.parameters)
        self.observation, _ = self.env.reset(
            seed=self.parameters["scenario"]["random_seed"]
        )

    def test_balloon_heights_are_above_sea_level(self):
        """Scenario 0 places balloon i at elevation + 10 + 40i."""
        count = int(self.parameters["balloon"]["num"])
        expected_above_pad = 10.0 + 40.0 * np.arange(count)
        np.testing.assert_allclose(
            self.env._balloon_flights[:, 2, 0] - self.elevation, expected_above_pad
        )
        # The same numbers reach the agent, not a pad-relative version of them.
        np.testing.assert_allclose(
            self.observation["balloon_states"][:, 2],
            expected_above_pad + self.elevation,
        )

    def test_horizontal_axes_are_measured_from_the_launch_point(self):
        launch = self._launch()
        np.testing.assert_allclose(
            np.asarray(launch[1:3], dtype=float), [0.0, 0.0], atol=1e-9
        )

    def test_the_rocket_starts_at_the_site_elevation(self):
        launch = self._launch()
        self.assertAlmostEqual(float(launch[3]), self.elevation, places=9)

    def test_gnss_reports_the_same_datum_as_the_true_state(self):
        self._launch()
        observation, _reward, _term, _trunc, info = self.env.step(self._idle())
        gnss_z = float(observation["rocket_sensors"][8])
        state_z = float(np.asarray(info["rocket_states"], dtype=float)[2])
        self.assertAlmostEqual(gnss_z, state_z, places=6)
        self.assertGreater(gnss_z, self.elevation - 1.0)

    def _idle(self):
        action = self.env.action_space.sample()
        action["launch"] = np.array(0, dtype=action["launch"].dtype)
        for key in ("tvc", "throttle", "roll"):
            action[key] = np.zeros_like(action[key])
        return action

    def _launch(self):
        action = self._idle()
        action["launch"] = np.array(1, dtype=action["launch"].dtype)
        action["launch_inclination_heading"] = np.array(
            [90.0, 0.0], dtype=action["launch_inclination_heading"].dtype
        )
        self.env.step(action)
        return self.env.initial_solution
