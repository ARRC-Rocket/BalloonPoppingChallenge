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

Cost / CI gating: this test is heavy, so it is gated behind the environment
variable ``BPC_RUN_SLOW_TESTS`` (the project has no pytest ``slow`` marker
infrastructure yet). The PR CI sets it, so the test runs there; a local run
without it is skipped. When the gate is on the simulation stack must import, so
an ImportError then fails loudly instead of skipping. Run it with::

    BPC_RUN_SLOW_TESTS=1 python -m pytest tests/test_scenario1_regression.py -v
"""

import json
import os
import unittest
from pathlib import Path

import numpy as np

# Only "1"/"true"/"yes" enable the slow gate; a leftover "0" or "false" does not.
_RUN_SLOW = os.environ.get("BPC_RUN_SLOW_TESTS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# ActiveRocketPy (imported as ``rocketpy``) is the heavy optional dependency. When
# the slow gate is on we intend to run, so a missing or broken stack must fail
# rather than skip. The BalloonPoppingGymEnv imports stay outside the guard: an
# ImportError from them is a real regression (a renamed or removed symbol) and
# must fail loudly when the stack is present.
try:
    import rocketpy  # noqa: F401
except ImportError:
    if _RUN_SLOW:
        raise
    _STACK_AVAILABLE = False
else:
    _STACK_AVAILABLE = True

    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

BASELINE_PATH = Path(__file__).parent / "baselines" / "scenario_1.json"

SCENARIO_NUMBER = 1

# Fixed agent configuration the baseline was generated with -- identical to
# scenario #0 (the agent flies the rocket straight up; the balloons drift past
# it on the wind, so no balloons are popped and the balloon ensemble is the
# interesting part of the run).
AGENT_KWARGS = {"rate_targets": [0.0, 0.0, 0.0], "launch_time": 1}

# The rocket trajectory is downsampled in time like scenario #0.
ROCKET_DOWNSAMPLE_STRIDE = 50
# The balloon array is (num_steps, num_balloons, 3). Downsample it in time only.
BALLOON_TIME_STRIDE = 200
# Keep every balloon (stride 1): sampling only 1 in N would leave the rest of the
# ensemble unchecked, so a regression in an unsampled balloon would pass
# unnoticed. Time downsampling alone keeps the baseline small (~30 points x 100).
BALLOON_INDEX_STRIDE = 1

# Scenario #1's fixed agent flies straight up and pops nothing; the balloons
# drift past on the wind. Pin that so regenerating cannot bless a different count.
EXPECTED_POPPED_COUNT = 0

# Per-coordinate position tolerance, applied as a real floor of
# ``max(POSITION_ATOL, POSITION_RTOL * abs(expected))`` -- not numpy's additive
# ``atol + rtol * expected`` -- so a coordinate near zero (the x/y balloon and
# rocket coordinates pass through 0) cannot drift by the metre floor *plus*
# another few percent.
POSITION_RTOL = 0.03
POSITION_ATOL = 1.0
# Flight duration is deterministic, so allow only a couple of steps of
# cross-platform jitter, absolute rather than a percentage: a 2% tolerance on a
# ~6000-step flight would wave through a full second of early termination.
STEP_COUNT_ABS_TOL = 2
ROW_COUNT_ABS_TOL = 1


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
    truncated = False
    while not (terminated or truncated):
        action = agent.get_action(observation)
        observation, _, terminated, truncated, _ = env.step(action)
    rocket_states = np.array(
        [step["rocket_states"] for step in env.trajectories], dtype=float
    )
    balloon_states = np.array(
        [step["balloon_states"] for step in env.trajectories], dtype=float
    )
    return rocket_states[:, :3], balloon_states[:, :, :3], int(env._popped_count)


def post_launch_rocket_positions(positions):
    """Return the launched rocket trajectory, downsampled to a clean ``(M, 3)``.

    The rows before launch are NaN, so launch is the first finite row. Every row
    from launch onwards must stay finite: a NaN or Inf appearing *after* launch
    means the flight diverged, so raise instead of dropping those rows (dropping
    them would let a broken tail slip past the comparison).
    """
    positions = np.asarray(positions, dtype=float)
    finite_rows = np.isfinite(positions).all(axis=1)
    launched = np.flatnonzero(finite_rows)
    if launched.size == 0:
        raise AssertionError("rocket never produced a finite post-launch position")

    post_launch = positions[launched[0] :]
    if not np.isfinite(post_launch).all():
        bad = np.flatnonzero(~np.isfinite(post_launch).all(axis=1))
        raise AssertionError(
            f"non-finite rocket position after launch at relative rows "
            f"{bad[:10].tolist()}"
        )
    return post_launch[::ROCKET_DOWNSAMPLE_STRIDE]


def downsample_balloon_positions(positions):
    """Downsample the balloon positions in time, keeping every balloon.

    ``positions`` is ``(num_steps, num_balloons, 3)``. Pre-release balloons are
    held at their initial Monte Carlo position (not NaN), so no NaN masking is
    needed; the slice is purely a size reduction along the time axis. Returns
    ``(T, num_balloons, 3)``.
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
        # Pin the semantic count, not just the (regenerable) baseline.
        self.assertEqual(self.baseline["popped_count"], EXPECTED_POPPED_COUNT)
        self.assertEqual(self.popped, EXPECTED_POPPED_COUNT)

    def test_flight_duration_matches_baseline(self):
        expected = self.baseline["num_steps_full"]
        self.assertLessEqual(
            abs(self.num_steps - expected),
            STEP_COUNT_ABS_TOL,
            f"flight step count {self.num_steps} drifted from baseline {expected} "
            f"by more than {STEP_COUNT_ABS_TOL} steps",
        )

    def test_rocket_position_trajectory_matches_baseline(self):
        expected = np.array(self.baseline["rocket_position_downsampled"], dtype=float)
        actual = self.rocket_positions
        self._assert_row_count_matches(len(actual), len(expected), "rocket")
        overlap = min(len(expected), len(actual))
        self._assert_within_floor(actual[:overlap], expected[:overlap], "rocket")

    def test_balloon_position_trajectory_matches_baseline(self):
        expected = np.array(self.baseline["balloon_position_downsampled"], dtype=float)
        actual = self.balloon_positions
        # Same balloon-index columns on both sides; only the time axis can drift
        # in length, so clip it to the common prefix (with the same row-count
        # guard as the rocket so an empty overlap cannot pass vacuously).
        self.assertEqual(actual.shape[1:], expected.shape[1:])
        self._assert_row_count_matches(actual.shape[0], expected.shape[0], "balloon")
        overlap = min(expected.shape[0], actual.shape[0])
        self._assert_within_floor(actual[:overlap], expected[:overlap], "balloon")

    def _assert_row_count_matches(self, actual_rows, expected_rows, label):
        self.assertLessEqual(
            abs(actual_rows - expected_rows),
            ROW_COUNT_ABS_TOL,
            f"{label} downsampled row count {actual_rows} drifted from baseline "
            f"{expected_rows} by more than {ROW_COUNT_ABS_TOL} rows "
            f"(possible launch failure or early termination)",
        )

    def _assert_within_floor(self, actual, expected, label):
        # Real floor: max(atol, rtol * |expected|), not numpy's additive form. A
        # NaN in ``actual`` propagates through ``np.max`` and fails the compare.
        error = np.abs(actual - expected)
        allowed = np.maximum(POSITION_ATOL, POSITION_RTOL * np.abs(expected))
        worst = float(np.max(error - allowed))
        self.assertLessEqual(
            worst,
            0.0,
            f"{label} position exceeds max({POSITION_ATOL} m, "
            f"{POSITION_RTOL:.0%} of |expected|) by {worst:.4g} m",
        )
        # Per-coordinate alone is not enough: three axes each 0.99 m off all pass
        # a 1 m floor while the point has actually moved 1.71 m, which is more than
        # the 1.5 m balloon radius the score depends on. Bound the displacement
        # vector as well, so the two together limit both a single axis relative to
        # its own magnitude and the total error.
        vector_error = np.linalg.norm(actual - expected, axis=-1)
        vector_allowed = np.maximum(
            POSITION_ATOL, POSITION_RTOL * np.linalg.norm(expected, axis=-1)
        )
        worst_vector = float(np.max(vector_error - vector_allowed))
        self.assertLessEqual(
            worst_vector,
            0.0,
            f"{label} 3D position error exceeds max({POSITION_ATOL} m, "
            f"{POSITION_RTOL:.0%} of |expected|) by {worst_vector:.4g} m",
        )


if __name__ == "__main__":
    unittest.main()


class TestTheToleranceShape(unittest.TestCase):
    """The per-coordinate floor alone allows more than the balloon radius.

    Pure arithmetic on the helper, so it needs no simulation stack and runs
    everywhere. Three axes each just inside a 1 m floor put the point 1.71 m away,
    and scenario 1's balloon radius is 1.5 m, so the score could move while the
    oracle stayed quiet.
    """

    def _helper(self):
        case = TestScenario1Regression("test_popped_count_matches_baseline")
        return case._assert_within_floor

    def test_three_axes_just_inside_the_floor_are_rejected(self):
        expected = np.zeros((1, 3))
        actual = np.full((1, 3), 0.99)

        # Each coordinate is within max(1.0, 3% of 0) = 1.0 m ...
        np.testing.assert_array_less(np.abs(actual - expected), POSITION_ATOL)
        # ... while the point has moved further than a balloon radius.
        self.assertGreater(float(np.linalg.norm(actual - expected)), 1.5)

        with self.assertRaises(AssertionError):
            self._helper()(actual, expected, "test")

    def test_a_displacement_inside_the_floor_is_accepted(self):
        expected = np.zeros((1, 3))
        actual = np.full((1, 3), 0.5)
        self.assertLess(float(np.linalg.norm(actual - expected)), POSITION_ATOL)

        self._helper()(actual, expected, "test")
