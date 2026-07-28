"""Behavioural test for issue #9 option (a): scenario 0 reset skips Monte Carlo.

Scenario 0 places balloons at fixed heights, so the Monte Carlo balloon
flight simulation is wasted work -- its result is fully overwritten with
static positions. reset() for scenario 0 must build the static balloon
array directly and never invoke MonteCarlo.

Runtime test: needs the simulation stack (numpy, rocketpy, ...). Skips
cleanly when the stack is not installed.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_0_PARAMS = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_0_parameters.yaml"
)

try:
    import numpy as np
    import yaml

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

    _STACK_AVAILABLE = True
except ImportError:
    _STACK_AVAILABLE = False

_MONTE_CARLO = "BalloonPoppingGymEnv.envs.balloon_world.MonteCarlo"


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestScenario0SkipsMonteCarlo(unittest.TestCase):
    """Issue #9 (a): scenario 0 builds static balloons without Monte Carlo."""

    def _make_scenario_0_env(self):
        with open(SCENARIO_0_PARAMS, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        return BalloonPoppingEnv(render_mode=None, parameters=params), params

    def test_reset_does_not_invoke_monte_carlo(self):
        env, _ = self._make_scenario_0_env()
        # MonteCarlo is booby-trapped: if scenario-0 reset touches it, it raises.
        with patch(_MONTE_CARLO, side_effect=AssertionError("MonteCarlo ran")):
            env.reset(seed=0)

    def test_reset_produces_static_balloons(self):
        env, params = self._make_scenario_0_env()
        with patch(_MONTE_CARLO, side_effect=AssertionError("MonteCarlo ran")):
            env.reset(seed=0)

        num = params["balloon"]["num"]
        flights = env._balloon_flights
        self.assertEqual(flights.shape[:2], (num, 6))

        # x, y, vx, vy, vz are zero for every balloon at every timestep
        self.assertTrue(np.all(flights[:, [0, 1, 3, 4, 5], :] == 0))

        # z is constant over time per balloon and matches the scenario-0 formula
        z = flights[:, 2, :]
        self.assertTrue(np.all(z == z[:, 0:1]))
        expected_z = 10 + env._rocketpy_env.elevation + np.arange(num) * 40
        np.testing.assert_allclose(z[:, 0], expected_z)

        # scenario 0 balloons all start released
        self.assertTrue(np.all(env._balloon_status == 1))


if __name__ == "__main__":
    unittest.main()
