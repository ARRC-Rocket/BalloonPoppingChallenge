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
from dataclasses import dataclass
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from tests.position_tolerance import (
    assert_launch_step_matches,
    assert_positions_match,
    displace_flight_in_time,
    launch_step,
)

# ActiveRocketPy (imported as ``rocketpy``) is the heavy optional dependency, so
# skip the whole test when it is genuinely not installed. The BalloonPoppingGymEnv
# imports are deliberately left outside that guard: once the stack is present, an
# ImportError from them is a real regression (a renamed or removed symbol, a
# broken internal import) and must fail the run instead of skipping it silently.
# ``find_spec`` answers "is the package installed", which is the only case that
# justifies a skip. The import itself stays outside any guard: ``import rocketpy``
# runs the package's own ``__init__``, so an ImportError raised there means an
# installed but broken stack, which is exactly what these tests exist to catch and
# must fail rather than skip.
_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import rocketpy  # noqa: F401

    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

BASELINE_PATH = Path(__file__).parent / "baselines" / "scenario_0.json"

# Fixed agent configuration the baseline was generated with.
AGENT_KWARGS = {"rate_targets": [0.0, 0.0, 0.0], "launch_time": 1}
DOWNSAMPLE_STRIDE = 50

# The position tolerance itself lives in tests/position_tolerance.py, shared
# with the other scenario so the two cannot drift apart again.
# Flight duration is deterministic, so allow only a couple of steps of
# cross-platform jitter. This is absolute, not a percentage: a 2% tolerance on a
# ~6000-step flight would wave through a full second of early termination.
STEP_COUNT_ABS_TOL = 2
# Downsampling turns those few steps into at most one row of difference.
ROW_COUNT_ABS_TOL = 1
# How far the flight is moved in the test that proves the launch step is being
# read. Ten steps is 0.1 s: five times STEP_COUNT_ABS_TOL, so it is not jitter,
# and a sixth of the 60 steps the trajectory comparison accepts on its own, so
# nothing but the launch step can be what rejects it.
DISPLACEMENT_PROBE_STEPS = 10
# Scenario #0 has 10 static balloons and the fixed agent is expected to pop them
# all; pin that semantic count so regenerating the baseline cannot quietly bless
# a regression that pops fewer.
SCENARIO_NUMBER = 0
EXPECTED_POPPED_COUNT = 10


@dataclass(frozen=True)
class RunResult:
    """One scenario-0 run, in the terms the baseline compares.

    ``positions`` is the per-step rocket centre-of-mass position
    ``(num_steps, 3)``, NaN rows before launch included.

    ``record_step`` is the simulation step each row was written at, so a row can
    be dated. Without it the trajectory comparison is purely launch-relative and
    the flight could have happened at any time; see ``launch_step``.

    ``pop_step`` is, for each balloon, the step at which it first reads as
    popped. Ten numbers, and the reason they are here: the final count alone
    says nothing about when or why anything popped. Measured against the
    previous version of this file, replacing the whole of ``_detect_pops`` with
    "mark every balloon popped on the first flown step" passed all three tests,
    because the count, the trajectory and the duration are all unchanged by it.

    ``terminated`` and ``truncated`` are kept apart because Gymnasium keeps them
    apart. Scenario 0's rocket lands well inside the horizon, so this run ends
    in a terminal state rather than at a limit, and swapping the two flags would
    otherwise be invisible.
    """

    positions: np.ndarray
    record_step: np.ndarray
    popped: int
    pop_step: np.ndarray
    status_history: np.ndarray
    terminated: bool
    truncated: bool


def run_scenario_0():
    """Run scenario #0 with the fixed agent and seed.

    Side effect free: it does not write trajectory files.
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
    # env.trajectories holds a per-step copy of the true rocket state.
    rocket_states = np.array(
        [step["rocket_states"] for step in env.trajectories], dtype=float
    )
    status_history = np.asarray(
        [step["balloon_status"] for step in env.trajectories], dtype=int
    )
    popped_now = status_history == 2
    has_popped = popped_now.any(axis=0)
    # A simulation step, not a row number. ``step()`` increments ``current_step``
    # before it appends the record and ``reset()`` appends nothing, so row i is
    # step i+1. Read the step back out of the record's own clock rather than
    # writing that offset down here, so this keeps meaning what its name says if
    # the logging ever starts somewhere else. The same offset is already carried
    # by scenario 1's release timing and by the submission checker. It dates the
    # pop steps below and the launch step the baseline is anchored to.
    record_step = np.rint(
        np.asarray([step["time"] for step in env.trajectories], dtype=float)
        / scenario_params["simulation"]["time_step"]
    ).astype(int)
    # -1 for a balloon that never popped, so a run that pops fewer is a visible
    # difference rather than a shorter list.
    pop_step = np.full(status_history.shape[1], -1, dtype=int)
    pop_step[has_popped] = record_step[popped_now.argmax(axis=0)[has_popped]]
    return RunResult(
        positions=rocket_states[:, :3],
        record_step=record_step,
        popped=int(env._popped_count),
        pop_step=pop_step,
        status_history=status_history,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def post_launch_positions(positions):
    """Return the launched trajectory, downsampled to a clean ``(M, 3)`` array.

    The rows before launch are NaN, so launch is the first finite row. Every row
    from launch onwards must stay finite: a NaN or Inf appearing *after* launch
    means the flight diverged, so raise instead of dropping those rows (dropping
    them would let a broken tail slip past the trajectory comparison).
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
    return post_launch[::DOWNSAMPLE_STRIDE]


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestScenario0Regression(unittest.TestCase):
    """Re-run scenario #0 once and compare it against the committed baseline."""

    @classmethod
    def setUpClass(cls):
        with open(BASELINE_PATH, encoding="utf-8") as baseline_file:
            cls.baseline = json.load(baseline_file)
        cls.run_result = run_scenario_0()
        cls.popped = cls.run_result.popped
        cls.num_steps = cls.run_result.positions.shape[0]
        cls.positions = post_launch_positions(cls.run_result.positions)
        cls.launch_step = launch_step(
            cls.run_result.positions, cls.run_result.record_step
        )

    def test_popped_count_matches_baseline(self):
        # Pin the semantic count, not just the (regenerable) baseline, so a
        # regression that pops fewer balloons cannot be blessed by regenerating.
        self.assertEqual(self.baseline["popped_count"], EXPECTED_POPPED_COUNT)
        self.assertEqual(self.popped, EXPECTED_POPPED_COUNT)

    def test_each_balloon_pops_when_the_baseline_says_it_does(self):
        """When and which, not only how many.

        The count alone is blind to the thing it exists to protect. Measured on
        the previous version: replacing the whole of ``_detect_pops`` with
        "mark every balloon popped on the first flown step" passed every test
        here, because it leaves the count at ten, the trajectory identical and
        the duration identical.

        Compared exactly rather than with a tolerance. These are step indices
        into a deterministic run, so a single step of difference is a real
        change to when a balloon was reached and worth looking at.

        Exact means exact on the platform the baseline was taken on. Two runs on
        one machine are bit identical and both CI Pythons agree, but a pop step
        is the first sample inside a radius, so a different libm or a different
        CPU path could in principle move one by a step. This is a golden master
        for the CI reference platform, not a claim about arbitrary hardware; if
        it ever fires on a new platform, check the neighbouring steps before
        concluding the geometry changed.
        """
        expected = np.asarray(self.baseline["pop_step"], dtype=int)
        actual = self.run_result.pop_step

        self.assertEqual(
            actual.shape,
            expected.shape,
            f"{actual.size} balloons, baseline has {expected.size}",
        )
        np.testing.assert_array_equal(actual, expected)

    def test_the_pops_are_spread_through_the_flight(self):
        """Guard the comparison above, which is against a regenerable file.

        If the baseline were regenerated from a run that popped everything at
        once, the exact comparison would keep passing forever. Ten static
        balloons stacked 40 m apart cannot all be reached in one step, so the
        steps have to differ.

        All ten distinct, rather than merely more than one. "More than one"
        rejects only the single mutation where every balloon pops together;
        nine on one step and one later would still pass, and so would a run
        that took them five at a time. The balloons are 40 m apart and the
        rocket covers about 1.4 m in a step, so a step that reaches two of them
        is not a tighter tolerance, it is a different simulation.
        """
        pop_step = self.run_result.pop_step

        self.assertTrue((pop_step >= 0).all(), "a balloon never popped")
        self.assertEqual(
            len(np.unique(pop_step)),
            EXPECTED_POPPED_COUNT,
            "two balloons popped on the same step, which 40 m of spacing forbids",
        )

    def test_no_balloon_goes_back_to_unpopped(self):
        """Only the first 2 is compared above, so the rest of the history is free.

        ``pop_step`` reads the first step each balloon shows as popped and the
        count reads the end state, so a status that went 1 -> 2 -> 1 -> 2 leaves
        both of them unchanged and every other test here passing. Scenario 1
        already pins this; the same hole was open on this side.
        """
        history = self.run_result.status_history

        backwards = np.argwhere(np.diff(history, axis=0) < 0)
        self.assertEqual(
            backwards.tolist()[:5],
            [],
            f"{len(backwards)} balloon status transitions go backwards, first "
            f"few at (step, balloon) {backwards.tolist()[:5]}",
        )

    def test_the_episode_ends_because_the_flight_ended(self):
        """Scenario 0 lands well inside the horizon.

        Both flags come back from ``step`` and were both discarded, so swapping
        them was invisible. Which one is set decides whether an agent should
        bootstrap from the final state.

        This discriminates because the two flags now carry different causes:
        ``terminated`` is the flight finishing and ``truncated`` is running out
        of precomputed horizon. While ``step`` still reported the clock as
        termination, these two assertions were also satisfied by an episode that
        simply ran out of time, and would not have established what the name
        says.
        """
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
            self.positions,
            np.array(self.baseline["rocket_position_downsampled"], dtype=float),
            "rocket",
            ROW_COUNT_ABS_TOL,
        )

    def test_the_flight_happens_when_the_baseline_says_it_does(self):
        """*When*, which the trajectory comparison throws away.

        ``post_launch_positions`` slices from the first finite row, so what is
        compared above is launch-relative and the same shape of flight passes
        wherever in the episode it sits.
        """
        assert_launch_step_matches(
            self, self.launch_step, self.baseline["launch_step"], STEP_COUNT_ABS_TOL
        )

    def test_a_flight_displaced_in_time_is_rejected(self):
        """The anchor above, pinned by the case that gets past everything else.

        Measured on this file before the anchor existed: the whole flight moved
        later by k steps, with the episode length preserved, passed all seven
        tests for every k up to 60. The first k to fail was 61, and it failed on
        the downsampled row count rather than on any clock.

        Both halves are asserted, because the pass is the point. The trajectory
        comparison is shown accepting the displaced flight, so the rejection
        below can only be coming from the launch step; if a later change made the
        trajectory comparison catch this on its own, the first assertion fails
        and says so instead of leaving a check here that nothing needs.
        """
        displaced = displace_flight_in_time(
            self.run_result.positions, DISPLACEMENT_PROBE_STEPS
        )

        assert_positions_match(
            self,
            post_launch_positions(displaced),
            np.array(self.baseline["rocket_position_downsampled"], dtype=float),
            "rocket",
            ROW_COUNT_ABS_TOL,
        )

        with self.assertRaises(AssertionError):
            assert_launch_step_matches(
                self,
                launch_step(displaced, self.run_result.record_step),
                self.baseline["launch_step"],
                STEP_COUNT_ABS_TOL,
            )


if __name__ == "__main__":
    unittest.main()
