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
from pathlib import Path

import numpy as np

# ActiveRocketPy (imported as ``rocketpy``) is the heavy optional dependency, so
# skip the whole test when it is genuinely not installed. The BalloonPoppingGymEnv
# imports are deliberately left outside that guard: once the stack is present, an
# ImportError from them is a real regression (a renamed or removed symbol, a
# broken internal import) and must fail the run instead of skipping it silently.
try:
    import rocketpy  # noqa: F401
except ImportError:
    _STACK_AVAILABLE = False
else:
    _STACK_AVAILABLE = True

    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

BASELINE_PATH = Path(__file__).parent / "baselines" / "scenario_0.json"

# Fixed agent configuration the baseline was generated with.
AGENT_KWARGS = {"rate_targets": [0.0, 0.0, 0.0], "launch_time": 1}
DOWNSAMPLE_STRIDE = 50

# Per-coordinate position tolerance, applied as a real floor of
# ``max(POSITION_ATOL, POSITION_RTOL * abs(expected))`` -- not numpy's additive
# ``atol + rtol * expected`` -- so a coordinate near zero cannot drift by the
# metre floor *plus* another few percent.
POSITION_RTOL = 0.03
POSITION_ATOL = 1.0
# Flight duration is deterministic, so allow only a couple of steps of
# cross-platform jitter. This is absolute, not a percentage: a 2% tolerance on a
# ~6000-step flight would wave through a full second of early termination.
STEP_COUNT_ABS_TOL = 2
# Downsampling turns those few steps into at most one row of difference.
ROW_COUNT_ABS_TOL = 1
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
    popped: int
    pop_step: np.ndarray
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
    # -1 for a balloon that never popped, so a run that pops fewer is a visible
    # difference rather than a shorter list.
    pop_step = np.where(popped_now.any(axis=0), popped_now.argmax(axis=0), -1)
    return RunResult(
        positions=rocket_states[:, :3],
        popped=int(env._popped_count),
        pop_step=pop_step,
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
        """
        pop_step = self.run_result.pop_step

        self.assertTrue((pop_step >= 0).all(), "a balloon never popped")
        self.assertGreater(
            len(np.unique(pop_step)),
            1,
            "every balloon popped on the same step, which the geometry forbids",
        )

    def test_the_episode_ends_because_the_flight_ended(self):
        """Scenario 0 lands well inside the horizon.

        Both flags come back from ``step`` and were both discarded, so swapping
        them was invisible. Which one is set decides whether an agent should
        bootstrap from the final state.
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
        expected = np.array(self.baseline["rocket_position_downsampled"], dtype=float)
        actual = self.positions
        # Guard the row count before comparing only the overlap: a truncated
        # trajectory (for example an early termination) would otherwise pass
        # vacuously on its shorter prefix.
        self.assertLessEqual(
            abs(len(actual) - len(expected)),
            ROW_COUNT_ABS_TOL,
            f"downsampled trajectory row count {len(actual)} drifted from "
            f"baseline {len(expected)} by more than {ROW_COUNT_ABS_TOL} rows",
        )
        overlap = min(len(expected), len(actual))
        # Real floor: max(atol, rtol * |expected|), not numpy's additive form.
        error = np.abs(actual[:overlap] - expected[:overlap])
        allowed = np.maximum(POSITION_ATOL, POSITION_RTOL * np.abs(expected[:overlap]))
        worst = float(np.max(error - allowed))
        self.assertLessEqual(
            worst,
            0.0,
            f"rocket position exceeds max({POSITION_ATOL} m, "
            f"{POSITION_RTOL:.0%} of |expected|) by {worst:.4g} m",
        )


if __name__ == "__main__":
    unittest.main()
