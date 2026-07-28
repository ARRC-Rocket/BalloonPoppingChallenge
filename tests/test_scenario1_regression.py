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
from typing import NamedTuple
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


class RunResult(NamedTuple):
    """What one scenario run reports, so callers stop unpacking a bare tuple."""

    rocket_positions: np.ndarray
    balloon_positions: np.ndarray
    popped_count: int
    final_status: np.ndarray
    status_counts: dict
    status_history: np.ndarray
    release_at_step: np.ndarray
    terminated: bool
    truncated: bool


def run_scenario_1():
    """Run scenario #1 with the fixed agent and seed, returning a `RunResult`.

    ``rocket_positions`` is the per-step rocket centre-of-dry-mass position
    ``(num_steps, 3)``, including the NaN rows before launch.
    ``balloon_positions`` is ``(num_steps, num_balloons, 3)``.
    ``popped_count`` is the final cumulative popped-balloon count.
    ``final_status`` is the last observation's balloon status, uncoerced.
    ``status_counts`` histograms it over ground, released and popped.
    ``status_history`` is ``(num_steps, num_balloons)``, so a caller can see
    *when* each balloon changed state, and ``release_at_step`` is the schedule it
    should match. ``terminated`` and ``truncated`` are the final Gymnasium flags,
    which mean different things and are both recorded rather than or-ed together.

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
    # Balloon status, because the sampled *positions* cannot see it: the balloon
    # positions come from the array precomputed and time-shifted at reset, so the
    # release machinery can stop working entirely without a single sampled
    # position moving. Deleting the ground-to-released update leaves all seven
    # position-and-count golden master tests green.
    #
    # Read it from the observation rather than ``env._balloon_status``. The two
    # hold the same values today, ``_get_obs`` forwards the array itself, but the
    # observation is what an agent is given, so a regression that stopped
    # forwarding the status would show up here instead of passing.
    #
    # Not coerced with ``dtype=int``: the env declares this observation as a
    # ``MultiDiscrete``, and a cast would quietly make a future non-integral
    # representation look valid before any assertion ran.
    final_status = np.asarray(observation["balloon_status"])[:, 0]
    status_counts = {
        "ground": int(np.sum(final_status == 0)),
        "released": int(np.sum(final_status == 1)),
        "popped": int(np.sum(final_status == 2)),
    }
    # ``step_record`` already carries ``balloon_status`` for every step, so the
    # release *timing* needs no extra instrumentation in the environment.
    status_history = np.asarray([step["balloon_status"] for step in env.trajectories])
    return RunResult(
        rocket_positions=rocket_states[:, :3],
        balloon_positions=balloon_states[:, :, :3],
        popped_count=int(env._popped_count),
        final_status=final_status,
        status_counts=status_counts,
        status_history=status_history,
        release_at_step=np.asarray(env._balloon_release_at_step, dtype=int),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


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
        cls.run_result = run_scenario_1()
        cls.popped = cls.run_result.popped_count
        cls.num_steps = cls.run_result.rocket_positions.shape[0]
        cls.rocket_positions = post_launch_rocket_positions(
            cls.run_result.rocket_positions
        )
        cls.balloon_positions = downsample_balloon_positions(
            cls.run_result.balloon_positions
        )

    def test_popped_count_matches_baseline(self):
        # Pin the semantic count, not just the (regenerable) baseline.
        self.assertEqual(self.baseline["popped_count"], EXPECTED_POPPED_COUNT)
        self.assertEqual(self.popped, EXPECTED_POPPED_COUNT)

    def test_the_release_and_pop_states_match_the_baseline(self):
        """The status machinery, which no sampled position can see.

        Balloon positions come from the array precomputed at reset, so removing
        the ground-to-released update in ``step`` leaves every sampled position,
        the step count and the expected score of zero exactly as they were. All
        seven golden master tests stay green with scenario 1's release machinery
        deleted, which this closes.

        The expected state is derived, not copied from the baseline. The last
        balloon is scheduled well before the episode ends, so the contract is
        that *every* balloon is accounted for: nothing still on the ground, and
        everything not popped released. Requiring only ``released + popped > 0``
        would have let a baseline regenerated against a partial failure, one
        balloon released out of a hundred, bless itself.
        """
        num_balloons = self.run_result.balloon_positions.shape[1]
        expected = {
            "ground": 0,
            "released": num_balloons - EXPECTED_POPPED_COUNT,
            "popped": EXPECTED_POPPED_COUNT,
        }

        self.assertEqual(self.run_result.status_counts, expected)
        self.assertEqual(self.baseline["final_balloon_status_counts"], expected)

    def test_the_balloon_status_stays_in_its_declared_domain(self):
        """The status observation is a ``MultiDiscrete``, so hold it to that.

        The histogram above only counts 0, 1 and 2. On its own that makes any
        other value invisible: it would be excluded from all three counts, and
        the counts would simply fail to add up with no indication why.
        """
        final_status = self.run_result.final_status

        self.assertTrue(
            np.issubdtype(final_status.dtype, np.integer),
            f"balloon status must be integral, got {final_status.dtype}",
        )
        unexpected = np.setdiff1d(np.unique(final_status), [0, 1, 2])
        self.assertEqual(
            unexpected.tolist(), [], "balloon status outside {ground, released, popped}"
        )
        self.assertEqual(
            sum(self.run_result.status_counts.values()),
            self.run_result.balloon_positions.shape[1],
            "the three status counts must account for every balloon",
        )
        # Two ways of counting the same thing, so they cannot drift apart.
        self.assertEqual(
            self.run_result.status_counts["popped"], self.run_result.popped_count
        )

    def test_every_balloon_is_released_at_its_scheduled_step(self):
        """*When* each balloon is released, not just how many end up released.

        The final histogram cannot see release timing. Releasing all hundred on
        the first step, or off-by-one on the release condition, reaches the same
        final ``0/100/0`` and leaves every sampled position untouched, because
        the flights were time-shifted at reset. ``step_record`` already stores
        ``balloon_status`` every step, so this needs nothing added to the
        environment.
        """
        history = self.run_result.status_history
        became_active = history != 0

        never_released = np.flatnonzero(~became_active.any(axis=0))
        self.assertEqual(
            never_released.tolist(), [], "balloons that never left the ground"
        )

        # `trajectories[i]` is written after `current_step` becomes `i + 1`, and
        # release fires on the first `current_step >= release_at_step`. One
        # balloon is scheduled at step 0, so its first recorded active step is 1.
        first_active_step = np.argmax(became_active, axis=0) + 1
        expected_first_active_step = np.maximum(self.run_result.release_at_step, 1)

        np.testing.assert_array_equal(first_active_step, expected_first_active_step)

    def test_the_release_schedule_itself_is_not_degenerate(self):
        """Guard the oracle above, which compares against a live attribute.

        If the schedule collapsed to all-zeros, "released when scheduled" would
        become "released immediately" and the comparison would still pass.
        """
        release_at_step = self.run_result.release_at_step
        num_steps = self.run_result.rocket_positions.shape[0]

        self.assertEqual(len(np.unique(release_at_step)), release_at_step.size)
        self.assertEqual(release_at_step.min(), 0)
        self.assertGreater(release_at_step.max(), 0)
        # Everything must be scheduled inside the episode, otherwise the
        # all-released contract above would be wrong rather than merely unmet.
        self.assertLess(release_at_step.max(), num_steps)

    def test_the_episode_terminates_rather_than_truncating(self):
        # Gymnasium gives these different meanings, and the runner only loops on
        # `terminated or truncated`, so swapping them keeps the step count intact.
        self.assertTrue(self.run_result.terminated)
        self.assertFalse(self.run_result.truncated)

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


if __name__ == "__main__":
    unittest.main()
