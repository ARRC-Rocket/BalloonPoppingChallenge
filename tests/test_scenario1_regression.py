"""Golden-master regression test for scenario #1 (issue #38).

Scenario #1 differs from scenario #0 in two ways that matter here. First, the
balloons are not static: on ``reset`` the environment runs a *Monte Carlo*
ensemble of 100 balloon flights through the ECMWF ensemble atmosphere
(``TW_Cup`` NetCDF), so the balloons drift and the scenario is comparatively
expensive (~30 s wall clock per full run). Second, because the balloons move,
the balloon trajectories themselves carry physics that scenario #0 cannot guard
-- so this test compares the (downsampled) balloon positions in addition to the
rocket trajectory.

Both the Monte Carlo balloon ensemble and the rocket integration are seeded from
``scenario.random_seed`` (0), so a fixed agent produces a reproducible run. This
test re-runs it and compares against a committed baseline, guarding against
unintended changes to the physics (for example when ActiveRocketPy is updated).

Determinism: two full runs with seed 0 on the same machine are bit-identical
(observed max position delta 0.0 for both rocket and balloons). The tolerances
below therefore only absorb cross-platform float drift; see the constants.

The baseline is tied to the current (repository, ActiveRocketPy submodule)
state. When a physics change is *intended*, regenerate it deliberately with
``tests/baselines/regenerate_scenario_1.py`` and review the diff.

Cost / CI gating: this test is opt-in. It only runs when the environment
variable ``BPC_RUN_SLOW_TESTS`` is set (the project has no pytest ``slow``
marker infrastructure yet), so it does not bloat the default PR CI. Run it
locally or in a nightly job with::

    BPC_RUN_SLOW_TESTS=1 python -m pytest tests/test_scenario1_regression.py -v
"""

import json
import os
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

_RUN_SLOW = bool(os.environ.get("BPC_RUN_SLOW_TESTS"))

BASELINE_PATH = Path(__file__).parent / "baselines" / "scenario_1.json"

SCENARIO_NUMBER = 1

# Fixed agent configuration the baseline was generated with -- identical to
# scenario #0 (the agent flies the rocket straight up; the balloons drift past
# it on the wind, so no balloons are popped and the balloon ensemble is the
# interesting part of the run).
AGENT_KWARGS = {"rate_targets": [0.0, 0.0, 0.0], "launch_time": 1}

# Rocket trajectory is downsampled like scenario #0. The balloon array is much
# larger -- (num_steps, num_balloons, 6) -- so it is downsampled in *both* time
# and balloon index to keep the committed baseline small and portable.
ROCKET_DOWNSAMPLE_STRIDE = 50
BALLOON_TIME_STRIDE = 200
BALLOON_INDEX_STRIDE = 10

# Same-machine runs are bit-identical; these tolerances absorb cross-platform
# float drift only (~3% per issue #38). atol is a metre floor for coordinates
# near zero (the x/y balloon and rocket coordinates pass through 0).
POSITION_RTOL = 0.03
POSITION_ATOL = 1.0
# A gross change in flight duration is a regression; a one-step cross-platform
# difference is not.
STEP_COUNT_REL_TOL = 0.02


def run_scenario_1():
    """Run scenario #1 with the fixed agent and seed.

    Returns ``(rocket_positions, balloon_positions, popped)`` where
    ``rocket_positions`` is the per-step rocket centre-of-mass position
    ``(num_steps, 3)`` -- including the NaN rows before launch -- and
    ``balloon_positions`` is the per-step balloon positions
    ``(num_steps, num_balloons, 3)``. ``popped`` is the final cumulative
    popped-balloon count.

    The env loop is driven directly (no ``evaluate.py``) so this is free of the
    ``save_trajectories`` file writes and the submission-MD5 network call. The
    only side effects are the Monte Carlo temp files ActiveRocketPy writes under
    the system temp dir on ``reset`` -- inherent to scenario #1 and unrelated to
    this test's output.
    """
    scenario_params, given_params = load_scenario_parameters(SCENARIO_NUMBER)
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
    agent = AttitudeRateControlAgent(given_params, **AGENT_KWARGS)
    observation, _ = env.reset(seed=scenario_params["scenario"]["random_seed"])
    terminated = False
    while not terminated:
        action = agent.get_action(observation)
        observation, _, terminated, _, _ = env.step(action)
    rocket_states = np.array(
        [step["rocket_states"] for step in env.trajectories], dtype=float
    )
    balloon_states = np.array(
        [step["balloon_states"] for step in env.trajectories], dtype=float
    )
    return rocket_states[:, :3], balloon_states[:, :, :3], int(env._popped_count)


def post_launch_rocket_positions(positions):
    """Drop the pre-launch NaN rows and downsample to a clean ``(M, 3)`` array."""
    launched = ~np.isnan(positions).any(axis=1)
    return positions[launched][::ROCKET_DOWNSAMPLE_STRIDE]


def downsample_balloon_positions(positions):
    """Downsample the balloon positions in time and balloon index.

    ``positions`` is ``(num_steps, num_balloons, 3)``. Pre-release balloons are
    held at their initial Monte Carlo position (not NaN), so no NaN masking is
    needed; the slice is purely a size reduction along fixed, run-stable indices.
    Returns ``(T, B, 3)``.
    """
    return positions[::BALLOON_TIME_STRIDE, ::BALLOON_INDEX_STRIDE, :]


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
@unittest.skipUnless(
    _RUN_SLOW, "slow Monte Carlo scenario; set BPC_RUN_SLOW_TESTS=1 to run"
)
class TestScenario1Regression(unittest.TestCase):
    """Re-run scenario #1 once and compare it against the committed baseline."""

    @classmethod
    def setUpClass(cls):
        with open(BASELINE_PATH, encoding="utf-8") as baseline_file:
            cls.baseline = json.load(baseline_file)
        rocket_positions, balloon_positions, cls.popped = run_scenario_1()
        cls.num_steps = rocket_positions.shape[0]
        cls.rocket_positions = post_launch_rocket_positions(rocket_positions)
        cls.balloon_positions = downsample_balloon_positions(balloon_positions)

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
        actual = self.rocket_positions
        # Guard the overlap clip below: a regression that stopped the rocket from
        # launching (or terminated it early) would leave the total step count
        # unchanged but collapse the post-launch row count, and an overlap of
        # zero rows would otherwise pass assert_allclose vacuously.
        self._assert_row_count_matches(len(actual), len(expected), "rocket")
        overlap = min(len(expected), len(actual))
        np.testing.assert_allclose(
            actual[:overlap],
            expected[:overlap],
            rtol=POSITION_RTOL,
            atol=POSITION_ATOL,
        )

    def test_balloon_position_trajectory_matches_baseline(self):
        expected = np.array(self.baseline["balloon_position_downsampled"], dtype=float)
        actual = self.balloon_positions
        # Same balloon-index columns on both sides; only the time axis can drift
        # in length, so clip it to the common prefix (with the same row-count
        # guard as the rocket so an empty overlap cannot pass vacuously).
        self.assertEqual(actual.shape[1:], expected.shape[1:])
        self._assert_row_count_matches(actual.shape[0], expected.shape[0], "balloon")
        overlap = min(expected.shape[0], actual.shape[0])
        np.testing.assert_allclose(
            actual[:overlap],
            expected[:overlap],
            rtol=POSITION_RTOL,
            atol=POSITION_ATOL,
        )

    def _assert_row_count_matches(self, actual_rows, expected_rows, label):
        drift = abs(actual_rows - expected_rows) / expected_rows
        self.assertLessEqual(
            drift,
            STEP_COUNT_REL_TOL,
            f"{label} downsampled row count {actual_rows} drifted from baseline "
            f"{expected_rows} (possible launch failure or early termination)",
        )


if __name__ == "__main__":
    unittest.main()
