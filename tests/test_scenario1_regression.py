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
from importlib.util import find_spec
from typing import NamedTuple
from pathlib import Path

import numpy as np

from tests.position_tolerance import (
    POSITION_VECTOR_ATOL,
    assert_launch_step_matches,
    assert_positions_match,
    displace_flight_in_time,
    launch_step,
)

# Only "1"/"true"/"yes" enable the slow gate; a leftover "0" or "false" does not.
_RUN_SLOW = os.environ.get("BPC_RUN_SLOW_TESTS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# ActiveRocketPy (imported as ``rocketpy``) is the heavy optional dependency.
# ``find_spec`` answers "is the package installed", which is the only case that
# justifies a skip. The import itself stays outside any guard: ``import rocketpy``
# runs the package's own ``__init__``, so an ImportError raised there means an
# installed but broken stack, which is exactly what these tests exist to catch
# and must fail rather than skip. The same goes for the BalloonPoppingGymEnv
# imports, where an ImportError means a renamed or removed symbol.
#
# And when the slow gate is on we intend to run, so a stack that is not there at
# all is an error too rather than a quiet pass over nothing.
_STACK_AVAILABLE = find_spec("rocketpy") is not None
if _RUN_SLOW and not _STACK_AVAILABLE:
    raise ImportError("BPC_RUN_SLOW_TESTS is set but rocketpy is not installed")

if _STACK_AVAILABLE:
    import rocketpy  # noqa: F401

    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import (
        BalloonPoppingEnv,
        _release_spacing_in_steps,
    )
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

# The position tolerance itself lives in tests/position_tolerance.py, shared
# with the other scenario so the two cannot drift apart again.
# Flight duration is deterministic, so allow only a couple of steps of
# cross-platform jitter, absolute rather than a percentage: a 2% tolerance on a
# ~6000-step flight would wave through a full second of early termination.
STEP_COUNT_ABS_TOL = 2
ROW_COUNT_ABS_TOL = 1
# How far the flight is moved in the test that proves the launch step is read.
# Ten steps is 0.1 s: five times STEP_COUNT_ABS_TOL, so not jitter, and a sixth
# of the 62 steps the trajectory comparison accepts on its own.
DISPLACEMENT_PROBE_STEPS = 10


class RunResult(NamedTuple):
    """What one scenario run reports, so callers stop unpacking a bare tuple."""

    rocket_positions: np.ndarray
    record_step: np.ndarray
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
    ``record_step`` dates each of those rows on the environment's own clock,
    which is what the rocket trajectory is otherwise missing: the comparison is
    launch-relative, while the balloons run on the absolute clock from step 0.
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
    # A simulation step, not a row number, read back off the record's own clock:
    # ``step()`` increments ``current_step`` before it appends the record, so row
    # i is step i+1. Scenario 0 dates its pop steps the same way.
    record_step = np.rint(
        np.asarray([step["time"] for step in env.trajectories], dtype=float)
        / scenario_params["simulation"]["time_step"]
    ).astype(int)
    return RunResult(
        rocket_positions=rocket_states[:, :3],
        record_step=record_step,
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
    # The shape, before the finite check. Both runners slice rocket_states to
    # its first three columns, so a state schema that lost one produces (T, 2)
    # rather than raising, and the regenerator would write that as the new
    # expected value. Same silent blessing the balloon helper refuses.
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise AssertionError(
            f"rocket positions must have shape (T, 3), got {positions.shape}"
        )
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

    A non-finite value here means a diverged balloon flight, so it raises rather
    than travelling into the comparison or, worse, into a regenerated baseline as
    a bare ``NaN`` token. The rocket helper already refuses one; this did not.

    The shape is checked as well, because a finite array is not necessarily a
    well formed one. ``(T, N, 2)`` from a state schema that lost a column, or an
    empty time or balloon axis, all pass the finite check and survive the slice
    unchanged. In a comparison the shape mismatch would surface, but this same
    helper feeds the *regenerator*, where a malformed array becomes the new
    expected value and the mismatch disappears with it.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or 0 in positions.shape[:2]:
        raise AssertionError(
            f"balloon positions must have shape (T, N, 3), got {positions.shape}"
        )
    if not np.isfinite(positions).all():
        raise AssertionError("balloon positions contain a non-finite value")
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
        cls.launch_step = launch_step(
            cls.run_result.rocket_positions, cls.run_result.record_step
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

    def test_every_recorded_status_is_one_of_the_three(self):
        """The whole history, not only the last row.

        The timing check above asks ``history != 0``, which treats any value at
        all as released, and the domain assertion elsewhere only looks at the
        final observation. So a history of ``0, 3, 1`` has the right first
        active step and the right final status while carrying a state that does
        not exist.
        """
        history = self.run_result.status_history

        self.assertTrue(
            np.issubdtype(history.dtype, np.integer),
            f"balloon status should be integral, got {history.dtype}",
        )
        outside = np.unique(history[~np.isin(history, (0, 1, 2))])
        self.assertEqual(
            outside.tolist(), [], "balloon status values outside 0, 1 and 2"
        )

    def test_no_balloon_goes_backwards(self):
        """A balloon on the ground is released, and a popped one stays popped.

        ``0, 1, 0, 1`` also has the right first active step and the right final
        histogram, so neither of the checks above sees it.
        """
        history = self.run_result.status_history

        backwards = np.argwhere(np.diff(history, axis=0) < 0)
        self.assertEqual(
            backwards[:5].tolist(),
            [],
            f"{len(backwards)} status transitions go backwards",
        )

    def test_the_release_schedule_matches_what_the_parameters_imply(self):
        """Derived from the scenario file, not read back from the environment:
        the old ``int(release_interval / time_step)`` repeated the environment's
        own division, comparing 2 against 2 where ``0.3 / 0.1`` should give 3."""
        parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        time_step = parameters["simulation"]["time_step"]
        release_interval = parameters["balloon"]["release_interval"]

        scheduled_seconds = np.sort(self.run_result.release_at_step) * time_step
        expected_seconds = np.arange(parameters["balloon"]["num"]) * release_interval

        # Half a time step: a correct schedule can only land on whole steps.
        np.testing.assert_allclose(
            scheduled_seconds, expected_seconds, rtol=0, atol=time_step / 2
        )

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
        assert_positions_match(
            self,
            self.rocket_positions,
            np.array(self.baseline["rocket_position_downsampled"], dtype=float),
            "rocket",
            ROW_COUNT_ABS_TOL,
        )

    def test_balloon_position_trajectory_matches_baseline(self):
        expected = np.array(self.baseline["balloon_position_downsampled"], dtype=float)
        actual = self.balloon_positions
        # Same balloon-index columns on both sides; only the time axis can drift
        # in length, and the helper clips that to the common prefix.
        self.assertEqual(actual.shape[1:], expected.shape[1:])

        assert_positions_match(self, actual, expected, "balloon", ROW_COUNT_ABS_TOL)

    # The lag between the step whose clock first reaches launch_time and the
    # first row carrying a finite state. Measured as 2 on both scenarios, and
    # structural: the action is read on one step, the state arrives on the next.
    LAUNCH_PIPELINE_LAG = 2

    def test_the_launch_step_is_the_one_the_agent_was_asked_for(self):
        """So regenerating cannot bless a flight that launches late: moving the
        agent's launch_time from 1 s to 3 s and regenerating writes launch_step
        302 with every test passing. Derived from the agent's own configuration."""
        parameters, _given = load_scenario_parameters(1)
        implied = AGENT_KWARGS["launch_time"] / parameters["simulation"]["time_step"]

        self.assertLessEqual(
            abs(self.baseline["launch_step"] - implied),
            STEP_COUNT_ABS_TOL + self.LAUNCH_PIPELINE_LAG,
            f"the baseline launches at step {self.baseline['launch_step']}, but a "
            f"launch_time of {AGENT_KWARGS['launch_time']} s implies about "
            f"{implied:.0f}",
        )

    def test_the_flight_happens_when_the_baseline_says_it_does(self):
        """*When*, which matters more here than on scenario 0. This run pops
        nothing, so nothing dates the rocket, while the balloons are compared on
        the absolute clock from step 0, and pop detection is their relative phase."""
        assert_launch_step_matches(
            self, self.launch_step, self.baseline["launch_step"], STEP_COUNT_ABS_TOL
        )

    def test_a_flight_displaced_in_time_is_rejected(self):
        """The anchor above, pinned by the case that gets past everything else:
        the rocket moved later by k steps, balloons left alone, passed all twelve
        tests for every k up to 62, and 63 failed on the downsampled row count."""
        displaced = displace_flight_in_time(
            self.run_result.rocket_positions, DISPLACEMENT_PROBE_STEPS
        )

        # Asserted to pass, so the rejection below can only be the launch step.
        assert_positions_match(
            self,
            post_launch_rocket_positions(displaced),
            np.array(self.baseline["rocket_position_downsampled"], dtype=float),
            "rocket",
            ROW_COUNT_ABS_TOL,
        )

        # Matched on the message, not just the type: ``launch_step`` raises
        # AssertionError for its own input guards too. Dropping the tail trim in
        # ``displace_flight_in_time`` left this passing on "6012 entries, 6022".
        with self.assertRaisesRegex(AssertionError, "launched at step"):
            assert_launch_step_matches(
                self,
                launch_step(displaced, self.run_result.record_step),
                self.baseline["launch_step"],
                STEP_COUNT_ABS_TOL,
            )


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheReleaseSpacing(unittest.TestCase):
    """The interval-to-steps conversion on its own, without running a scenario.
    The shipped numbers divide exactly, so the error this catches is invisible in
    the scenario test above; these call it on intervals the scenarios never use."""

    def test_an_interval_that_divides_just_under_a_whole_step(self):
        """Both operands are float seconds, so their quotient is a binary float:
        0.3 / 0.1 is 2.9999999999999996, 0.7 / 0.1 is 6.999999999999999 and
        2.3 / 0.1 is 22.999999999999996. Truncating any of them loses a step."""
        for release_interval, time_step, expected in (
            (0.3, 0.1, 3),
            (0.7, 0.1, 7),
            (2.3, 0.1, 23),
        ):
            with self.subTest(release_interval=release_interval):
                self.assertEqual(
                    _release_spacing_in_steps(release_interval, time_step), expected
                )

    def test_the_shipped_scenarios_keep_the_spacing_they_already_had(self):
        """Rounding instead of truncating must not move a committed baseline.
        1 / 0.01 and 0.5 / 0.01 are exactly integral, so the two scenarios are
        unaffected; pinned because a scenario edit is when that would change."""
        for scenario, expected in ((0, 100), (SCENARIO_NUMBER, 50)):
            with self.subTest(scenario=scenario):
                parameters, _ = load_scenario_parameters(scenario)

                self.assertEqual(
                    _release_spacing_in_steps(
                        parameters["balloon"]["release_interval"],
                        parameters["simulation"]["time_step"],
                    ),
                    expected,
                )


class TestTheToleranceShape(unittest.TestCase):
    """The comparison helper itself, on made-up numbers so it runs anywhere.

    These live here rather than only inside the slow scenario tests, which need
    the whole Monte Carlo stack and a baseline before they can say anything about
    the shape of the tolerance.
    """

    def _compare(self, actual, expected, row_count_abs_tol=1):
        assert_positions_match(
            self,
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            "test",
            row_count_abs_tol,
        )

    def test_three_axes_inside_the_floor_can_still_move_the_point_too_far(self):
        # The case the per-coordinate check alone cannot see: each axis is 0.99 m
        # off, inside the 1 m floor, while the point has moved 1.71 m.
        expected = [[0.0, 0.0, 0.0]]
        actual = [[0.99, 0.99, 0.99]]

        with self.assertRaises(AssertionError):
            self._compare(actual, expected)

    def test_the_cap_does_not_grow_with_altitude(self):
        """The reason this cap is absolute.

        Z is altitude above sea level, so a relative bound scales with distance
        from sea level rather than with anything about the trajectory. At 1000 m
        a 3% bound would allow 30 m, and on the committed baseline it reaches
        48.8 m at apogee while the balloon radius is 1.5 m.
        """
        expected = [[0.0, 0.0, 1000.0]]
        actual = [[0.0, 0.0, 1000.0 + POSITION_VECTOR_ATOL + 0.01]]

        with self.assertRaises(AssertionError):
            self._compare(actual, expected)

    def test_a_displacement_inside_the_cap_passes_at_altitude(self):
        expected = [[0.0, 0.0, 1000.0]]
        actual = [[0.0, 0.0, 1000.0 + POSITION_VECTOR_ATOL - 0.01]]

        self._compare(actual, expected)

    def test_a_non_finite_value_in_a_tolerated_extra_row_fails(self):
        """The row the count tolerance allows is still data.

        Checking after the overlap slice discarded it, so a diverged tail in the
        one extra row passed: exactly the case a golden master exists to catch.
        """
        expected = [[0.0, 0.0, 10.0]] * 5
        actual = [[0.0, 0.0, 10.0]] * 5 + [[0.0, 0.0, float("nan")]]

        with self.assertRaisesRegex(AssertionError, "non-finite"):
            self._compare(actual, expected)

    def test_a_baseline_that_lost_coordinates_cannot_broadcast(self):
        """(T, 1) against (T, 3) is a valid NumPy subtraction, not an error.

        The trailing axis of length one is stretched to three, the displacement
        comes out zero, and a baseline missing Y and Z passes.
        """
        with self.assertRaisesRegex(AssertionError, "XYZ|shape"):
            self._compare(np.zeros((4, 3)), np.zeros((4, 1)))

    def test_both_sides_losing_z_is_not_a_two_dimensional_oracle(self):
        """A regenerated baseline degrades in step with the run.

        Both at (T, 2) subtracts and takes a norm without complaint, and the
        comparison silently stops covering the third axis.
        """
        with self.assertRaisesRegex(AssertionError, "XYZ"):
            self._compare(np.zeros((4, 2)), np.zeros((4, 2)))

    def test_an_empty_balloon_axis_is_an_assertion(self):
        # np.max over an empty result raises ValueError, which reads as a broken
        # test rather than a trajectory with no balloons in it.
        with self.assertRaises(AssertionError):
            self._compare(np.zeros((4, 0, 3)), np.zeros((4, 0, 3)))

    def test_a_shape_that_disagrees_with_the_baseline_fails(self):
        with self.assertRaisesRegex(AssertionError, "does not match baseline"):
            self._compare(np.zeros((4, 2, 3)), np.zeros((4, 3, 3)))

    def test_a_non_finite_value_in_the_baseline_fails_too(self):
        """Both sides, not only the fresh run.

        A baseline regenerated while something was diverging would otherwise be
        compared against happily.

        The bad row is the extra one, and the message is pinned. Put it inside
        the overlap and the displacement comes out infinite, so the cap check
        raises AssertionError and a test that only asked for AssertionError
        could not tell which check had fired. Measured: with the baseline left
        unchecked, that version still passed.
        """
        expected = [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0], [0.0, 0.0, float("inf")]]
        actual = [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]]

        with self.assertRaisesRegex(AssertionError, "baseline.*non-finite"):
            self._compare(actual, expected)

    def test_an_empty_overlap_is_an_assertion_not_a_crash(self):
        """A row-count difference inside the tolerance can still leave nothing.

        np.max over an empty array raises ValueError, which reads as a broken
        test rather than a failed comparison.
        """
        with self.assertRaises(AssertionError):
            self._compare(np.zeros((0, 3)), [[0.0, 0.0, 10.0]])

    def test_a_non_finite_value_fails_rather_than_being_skipped(self):
        """NaN comparisons are all false, so a check can pass by ignoring them.

        A diverged trajectory is the case this whole comparison exists for, and
        reaching for nanmax to keep the numbers tidy would let it through.
        """
        expected = [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]]
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad):
                actual = [[0.0, 0.0, 10.0], [0.0, 0.0, bad]]

                with self.assertRaises(AssertionError):
                    self._compare(actual, expected)

    def test_a_tolerated_row_count_difference_does_not_raise(self):
        """The one-row drift the count tolerance exists to allow.

        Subtracting the full arrays raises a broadcasting ValueError here rather
        than comparing anything, so the tolerance silently did not work.
        """
        expected = [[0.0, 0.0, 10.0]] * 5
        actual = [[0.0, 0.0, 10.0]] * 6

        self._compare(actual, expected)

    def test_a_row_count_difference_past_the_tolerance_fails(self):
        expected = [[0.0, 0.0, 10.0]] * 5
        actual = [[0.0, 0.0, 10.0]] * 8

        with self.assertRaises(AssertionError):
            self._compare(actual, expected)

    def test_the_helper_works_on_the_balloon_array_shape(self):
        expected = np.zeros((4, 3, 3))
        actual = expected.copy()
        actual[2, 1, 2] = POSITION_VECTOR_ATOL + 0.01

        with self.assertRaises(AssertionError):
            self._compare(actual, expected)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheCapStaysBelowTheRadius(unittest.TestCase):
    """The cap is only meaningful while it is well under the balloon radius.

    Read from the scenario rather than written as 1.5 here, so raising the
    radius in the YAML cannot leave this claim quietly false.
    """

    def test_the_combined_worst_case_stays_under_the_balloon_radius(self):
        """Twice the cap, because scenario 1 applies it to two trajectories.

        The rocket and the balloons are compared separately, each against the
        same cap. Pop detection depends on the distance between them, so two
        independent errors of the full cap in opposite directions move that
        distance by twice the cap. Comparing a single cap against the radius was
        the wrong figure: at a metre the combined worst case was 2 m against a
        1.5 m radius.
        """
        for scenario in (0, SCENARIO_NUMBER):
            with self.subTest(scenario=scenario):
                parameters, _ = load_scenario_parameters(scenario)

                self.assertLess(
                    2 * POSITION_VECTOR_ATOL, parameters["balloon"]["radius"]
                )


if __name__ == "__main__":
    unittest.main()
