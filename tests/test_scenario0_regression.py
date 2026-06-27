"""Golden-master regression test for scenario #0 (issue #38).

Scenario #0 is deterministic -- static balloons, a fixed seed, ideal sensors and
no Monte Carlo -- so a fixed agent produces a reproducible rocket trajectory.
This test re-runs it and compares against a committed baseline, guarding against
unintended changes to the physics (for example when ActiveRocketPy is updated).

The baseline is tied to the current (repository, ActiveRocketPy submodule) state.
When a physics change is *intended*, regenerate it deliberately with
``tests/baselines/regenerate_scenario_0.py`` and review the diff.

Scope (v1): the popped-balloon count (exact) and the rocket position trajectory
(~3% tolerance). Attitude and velocity are intentionally left for a later
iteration to keep this guard small and its tolerances simple.
"""

import json
import unittest
from pathlib import Path

try:
    import numpy as np

    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

    _STACK_AVAILABLE = True
except ImportError:
    _STACK_AVAILABLE = False

BASELINE_PATH = Path(__file__).parent / "baselines" / "scenario_0.json"

# Fixed agent configuration the baseline was generated with.
AGENT_KWARGS = {"rate_targets": [0.0, 0.0, 0.0], "launch_time": 1}
DOWNSAMPLE_STRIDE = 50

# Same-machine runs are bit-identical; these tolerances absorb cross-platform
# float drift only (~3% per issue #38). atol is a metre floor for coordinates
# near zero.
POSITION_RTOL = 0.03
POSITION_ATOL = 1.0
# A gross change in flight duration is a regression; a one-step cross-platform
# difference is not.
STEP_COUNT_REL_TOL = 0.02


def run_scenario_0():
    """Run scenario #0 with the fixed agent and seed.

    Returns ``(positions, popped)`` where ``positions`` is the per-step rocket
    centre-of-mass position ``(num_steps, 3)`` -- including the NaN rows before
    launch -- and ``popped`` is the final cumulative popped-balloon count. Side
    effect free: it does not write trajectory files.
    """
    scenario_params, given_params = load_scenario_parameters(0)
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
    agent = AttitudeRateControlAgent(given_params, **AGENT_KWARGS)
    observation, _ = env.reset(seed=scenario_params["scenario"]["random_seed"])
    terminated = False
    while not terminated:
        action = agent.get_action(observation)
        observation, _, terminated, _, _ = env.step(action)
    # env.trajectories holds a per-step copy of the true rocket state.
    rocket_states = np.array(
        [step["rocket_states"] for step in env.trajectories], dtype=float
    )
    return rocket_states[:, :3], int(env._popped_count)


def post_launch_positions(positions):
    """Drop the pre-launch NaN rows and downsample to a clean ``(M, 3)`` array."""
    launched = ~np.isnan(positions).any(axis=1)
    return positions[launched][::DOWNSAMPLE_STRIDE]


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestScenario0Regression(unittest.TestCase):
    """Re-run scenario #0 once and compare it against the committed baseline."""

    @classmethod
    def setUpClass(cls):
        with open(BASELINE_PATH, encoding="utf-8") as baseline_file:
            cls.baseline = json.load(baseline_file)
        positions, cls.popped = run_scenario_0()
        cls.num_steps = positions.shape[0]
        cls.positions = post_launch_positions(positions)

    def test_popped_count_matches_baseline(self):
        self.assertEqual(self.popped, self.baseline["popped_count"])

    def test_flight_duration_matches_baseline(self):
        expected = self.baseline["num_steps_full"]
        drift = abs(self.num_steps - expected) / expected
        self.assertLessEqual(
            drift,
            STEP_COUNT_REL_TOL,
            f"flight step count {self.num_steps} drifted from baseline {expected}",
        )

    def test_rocket_position_trajectory_matches_baseline(self):
        expected = np.array(self.baseline["rocket_position_downsampled"], dtype=float)
        actual = self.positions
        overlap = min(len(expected), len(actual))
        np.testing.assert_allclose(
            actual[:overlap],
            expected[:overlap],
            rtol=POSITION_RTOL,
            atol=POSITION_ATOL,
        )


if __name__ == "__main__":
    unittest.main()
