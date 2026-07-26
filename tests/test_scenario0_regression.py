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


def run_scenario_0():
    """Run scenario #0 with the fixed agent and seed.

    Returns ``(positions, popped)`` where ``positions`` is the per-step rocket
    centre-of-mass position ``(num_steps, 3)`` -- including the NaN rows before
    launch -- and ``popped`` is the final cumulative popped-balloon count. Side
    effect free: it does not write trajectory files.
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
    return rocket_states[:, :3], int(env._popped_count)


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
        positions, cls.popped = run_scenario_0()
        cls.num_steps = positions.shape[0]
        cls.positions = post_launch_positions(positions)

    def test_popped_count_matches_baseline(self):
        # Pin the semantic count, not just the (regenerable) baseline, so a
        # regression that pops fewer balloons cannot be blessed by regenerating.
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
