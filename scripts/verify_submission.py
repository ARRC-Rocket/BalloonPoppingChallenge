#!/usr/bin/env python3
"""Check a submission's balloon trajectories against what its seed produces.

The balloons are not something a competitor controls. Their flights come out of
the scenario parameters and the seed, and an agent only ever commands the
rocket, so regenerating them from the submission's own
``scenario_parameters`` gives an independent copy of what the run should have
recorded. Anything that does not match was not produced by the simulator.

Run it against a submission downloaded from the leaderboard::

    uv run python scripts/verify_submission.py 20260728T115315Z_team_submission.pkl

Exit status is 0 when every check passes and 1 otherwise, so it can be used in
a loop over a directory of submissions.

What this does not do
---------------------
It does not verify the rocket trajectory, and the score comes from the rocket
trajectory. Checking that means replaying the agent, whose source is sitting in
the submission, and running a competitor's code server side is arbitrary code
execution by design. That was decided against in the leaderboard's issue #4.

So a submission that passes has balloons the simulator would have produced,
and a rocket path that is still the competitor's claim. What it catches is the
straightforward edit: moving balloons onto the rocket's real path to claim pops
that never happened, or rewriting the score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ``trajectories[k]["balloon_states"]`` is ``_balloon_flights[:, :, k + 1]``,
# not ``[:, :, k]``. ``step()`` advances ``current_step``, reads the flight
# array at that index, and only then appends the record.
#
# Getting this wrong does not look like an off-by-one, it looks like tampering.
# Measured on a real scenario-1 submission: offset 1 matches to exactly zero,
# offset 0 and offset 2 both differ by 2.80 m, which is larger than a balloon.
# Scenario 0 cannot tell you this, because its balloons are static and its
# flight array is the same at every timestep, so any offset passes.
_TRAJECTORY_OFFSET = 1

# Two runs of the same seed on the same machine reproduce each other exactly, so
# the honest default is "no difference at all, give or take arithmetic". A
# different machine will not be exact, which is why the report prints the
# measured difference rather than only a verdict: 1e-9 m is a different BLAS,
# and anything approaching the balloon radius is a different trajectory.
DEFAULT_TOLERANCE_METRES = 1e-6


class Finding:
    """One thing that was checked, and what it showed."""

    def __init__(self, name, ok, detail):
        self.name = name
        self.ok = ok
        self.detail = detail

    def __str__(self):
        return f"[{'ok  ' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def load_submission(path):
    """Read a submission, whichever container it is in.

    Submissions are pickle today and JSON once the format change lands. Reading
    by suffix means this keeps working across that without a second copy.

    The pickle is loaded with the restricted unpickler the leaderboard uses if
    it is importable, and refused otherwise: a submission being checked for
    tampering is exactly the file you do not want to unpickle blindly.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    import pickle

    allowed = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
    }

    class _Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in allowed:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"submission refers to {module}.{name}, which a submission does "
                "not contain; refusing to load it"
            )

    with open(path, "rb") as handle:
        return _Restricted(handle).load()


def _regenerate_balloon_flights(scenario_parameters):
    """Run the environment's own reset to get the flights the seed implies.

    Deliberately the real environment rather than a reimplementation of the
    Monte Carlo. A checker whose physics can drift from the simulator's is worse
    than no checker, because a disagreement then says nothing about which one is
    wrong.
    """
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)
    env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    return np.asarray(env._balloon_flights, dtype=float), np.asarray(
        env._balloon_release_at_step
    )


def check_balloon_trajectories(submission, tolerance, trusted_positions):
    """The main check: every recorded balloon state came out of the simulator.

    ``trusted_positions`` is appended to with the regenerated positions, so the
    pop check below can reuse them instead of paying for the Monte Carlo twice.
    """
    findings = []
    world = submission["balloon_world_data"]
    records = world["trajectories"]

    claimed = np.asarray([record["balloon_states"] for record in records], dtype=float)
    if claimed.ndim != 3 or claimed.shape[2] != 6:
        return [
            Finding(
                "balloon states",
                False,
                f"expected (steps, balloons, 6), got {claimed.shape}",
            )
        ]

    flights, release_at_step = _regenerate_balloon_flights(world["scenario_parameters"])

    expected_balloons = flights.shape[0]
    if claimed.shape[1] != expected_balloons:
        findings.append(
            Finding(
                "balloon count",
                False,
                f"submission records {claimed.shape[1]}, the scenario has "
                f"{expected_balloons}",
            )
        )
        return findings
    findings.append(
        Finding("balloon count", True, f"{expected_balloons} balloons, as expected")
    )

    steps = claimed.shape[0]
    available = flights.shape[2] - _TRAJECTORY_OFFSET
    if steps > available:
        findings.append(
            Finding(
                "episode length",
                False,
                f"submission has {steps} steps, the scenario can only produce "
                f"{available}",
            )
        )
        return findings

    expected = np.transpose(
        flights[:, :, _TRAJECTORY_OFFSET : _TRAJECTORY_OFFSET + steps], (2, 0, 1)
    )

    if not np.isfinite(claimed).all():
        findings.append(
            Finding(
                "balloon states are numbers",
                False,
                "the submission contains non-finite balloon states",
            )
        )
        return findings

    # Positions only. The velocities come from the same integration and add
    # nothing a position difference would not already show, and mixing metres
    # per second into a metre tolerance would make the number meaningless.
    trusted_positions.append(expected[:, :, :3])
    difference = np.abs(expected[:, :, :3] - claimed[:, :, :3])
    worst = float(difference.max())
    findings.append(
        Finding(
            "balloon trajectories",
            worst <= tolerance,
            f"largest position difference {worst:.3e} m over {steps} steps "
            f"(tolerance {tolerance:.0e} m)",
        )
    )

    if worst > tolerance:
        # Say where, so the difference can be looked at rather than argued about.
        per_balloon = difference.max(axis=(0, 2))
        order = np.argsort(per_balloon)[::-1][:5]
        for index in order:
            if per_balloon[index] <= tolerance:
                continue
            first = int(np.argmax(difference[:, index, :].max(axis=1) > tolerance))
            findings.append(
                Finding(
                    f"balloon {index}",
                    False,
                    f"differs by up to {per_balloon[index]:.3e} m, first at step "
                    f"{first} (t={records[first]['time']:.3f} s)",
                )
            )

    claimed_release = np.asarray(world["balloon_release_at_step"])
    findings.append(
        Finding(
            "release schedule",
            np.array_equal(claimed_release, release_at_step),
            "matches the schedule the seed produces"
            if np.array_equal(claimed_release, release_at_step)
            else "does not match the schedule the seed produces",
        )
    )
    return findings


def check_claimed_pops_are_reachable(submission, expected_positions):
    """Every balloon claimed popped has to have been somewhere the rocket went.

    The checks above catch a balloon that was moved. They do not catch the
    easier edit: leave every position alone, flip a released balloon's status to
    2 at a moment that looks legal, and raise the score to match. Measured
    against an earlier version of this script, twenty fabricated pops passed
    every other check here.

    This uses the trusted balloon positions with the rocket path the submission
    claims. The rocket path is not verifiable without replaying the agent, which
    is out of scope, but fabricating a continuous rocket trajectory that happens
    to pass through the balloons you want is a different order of work from
    editing an integer.

    Deliberately reports the closest approach rather than reproducing the pop
    decision. Reproducing it means matching the environment's sweep boundaries
    exactly, including the one interval whose origin is the launch state rather
    than a recorded position, and being subtly wrong there would accuse an
    honest competitor. A boundary that is one step out moves the distance by
    about a metre; a fabricated pop is out by hundreds.
    """
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

    world = submission["balloon_world_data"]
    records = world["trajectories"]
    radius = float(world["scenario_parameters"]["balloon"]["radius"])

    status = np.asarray([record["balloon_status"] for record in records], dtype=int)
    claimed_popped = np.flatnonzero((status == 2).any(axis=0))
    if claimed_popped.size == 0:
        return [Finding("claimed pops", True, "the submission claims no pops")]

    rocket = np.asarray(
        [record["rocket_states"][:3] for record in records], dtype=float
    )
    flyable = np.isfinite(rocket).all(axis=1)
    if flyable.sum() < 2:
        return [
            Finding(
                "claimed pops",
                False,
                f"{claimed_popped.size} balloons are claimed popped, but the "
                "submission records no rocket positions at all",
            )
        ]

    balloons = expected_positions[:, claimed_popped, :]
    closest = np.full(claimed_popped.size, np.inf)
    # Only intervals where both ends of the rocket segment are real. The
    # pre-launch rows are all-NaN by construction.
    steps = np.flatnonzero(flyable[:-1] & flyable[1:])
    for step in steps:
        released = status[step, claimed_popped] >= 1
        if not released.any():
            continue
        distance_squared = BalloonPoppingEnv._segment_distance_squared_batch(
            rocket[step],
            rocket[step + 1],
            balloons[step][released],
            balloons[step + 1][released],
        )
        candidate = np.sqrt(np.asarray(distance_squared, dtype=float))
        where = np.flatnonzero(released)
        closest[where] = np.minimum(closest[where], candidate)

    unreachable = closest > radius
    findings = [
        Finding(
            "claimed pops are reachable",
            not unreachable.any(),
            f"{claimed_popped.size} balloons claimed popped, closest approach "
            f"{np.nanmax(closest[np.isfinite(closest)]) if np.isfinite(closest).any() else float('inf'):.3f} m "
            f"at worst against a {radius} m radius",
        )
    ]
    for position in np.flatnonzero(unreachable)[:5]:
        distance = closest[position]
        findings.append(
            Finding(
                f"balloon {claimed_popped[position]}",
                False,
                "claimed popped, but the rocket path never came closer than "
                + (
                    f"{distance:.1f} m"
                    if np.isfinite(distance)
                    else "any measurable distance, since it was never released "
                    "while the rocket was flying"
                ),
            )
        )
    return findings


def check_internal_consistency(submission):
    """Cheap checks that need no physics, so they cannot disagree with any.

    The trajectory check above says the balloons are real. It says nothing about
    whether the score matches the run, and the simplest edit of all is to raise
    ``final_reward`` and leave everything else alone.
    """
    findings = []
    world = submission["balloon_world_data"]
    records = world["trajectories"]
    status = np.asarray([record["balloon_status"] for record in records], dtype=int)

    claimed_score = submission["leaderboard_info"]["final_reward"]
    implied = int((status[-1] == 2).sum())
    findings.append(
        Finding(
            "score matches the record",
            float(claimed_score) == float(implied),
            f"submission claims {claimed_score}, the final statuses show {implied}",
        )
    )

    # 0 on the ground, 1 released, 2 popped. A balloon never goes back.
    went_backwards = int((np.diff(status, axis=0) < 0).sum())
    findings.append(
        Finding(
            "balloon status only moves forward",
            went_backwards == 0,
            "no balloon returns to an earlier state"
            if went_backwards == 0
            else f"{went_backwards} status transitions go backwards",
        )
    )

    # A balloon on the ground cannot be popped, so 2 never follows 0 directly.
    skipped = int(((status[:-1] == 0) & (status[1:] == 2)).sum())
    findings.append(
        Finding(
            "no balloon is popped before release",
            skipped == 0,
            "every pop follows a release"
            if skipped == 0
            else f"{skipped} balloons go from the ground straight to popped",
        )
    )
    return findings


def verify(submission, tolerance):
    for key in ("balloon_world_data", "leaderboard_info"):
        if key not in submission:
            return [Finding("structure", False, f"submission has no {key!r}")]
    world = submission["balloon_world_data"]
    for key in ("trajectories", "scenario_parameters", "balloon_release_at_step"):
        if key not in world:
            return [Finding("structure", False, f"balloon_world_data has no {key!r}")]
    if not world["trajectories"]:
        return [Finding("structure", False, "the submission records no steps")]

    trusted_positions = []
    findings = check_balloon_trajectories(submission, tolerance, trusted_positions)
    findings += check_internal_consistency(submission)
    # Only worth asking where the rocket went if the balloons it is compared
    # against are the ones the seed produced.
    if trusted_positions and all(finding.ok for finding in findings):
        findings += check_claimed_pops_are_reachable(submission, trusted_positions[0])
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("submission", nargs="+", help="submission file(s) to check")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_METRES,
        help=(
            "largest position difference to accept, in metres. The default "
            f"({DEFAULT_TOLERANCE_METRES:.0e}) is tight because the same seed on "
            "the same machine reproduces exactly. Raise it if you are checking "
            "on different hardware from the one that produced the run, and read "
            "the reported difference rather than only the verdict: a balloon is "
            "1.5 m across, so a difference approaching that is a different "
            "trajectory rather than different arithmetic."
        ),
    )
    args = parser.parse_args(argv)

    failed = False
    for path in args.submission:
        print(f"\n{os.path.basename(path)}")
        try:
            submission = load_submission(path)
        except Exception as exc:  # noqa: BLE001 - report and move to the next file
            print(f"  [FAIL] could not read it: {exc}")
            failed = True
            continue

        info = submission.get("leaderboard_info", {})
        print(
            f"  team {info.get('team_name')!r}, agent {info.get('agent_name')!r}, "
            f"scenario {info.get('scenario_number')}, claimed score "
            f"{info.get('final_reward')}"
        )
        findings = verify(submission, args.tolerance)
        for finding in findings:
            print(f"  {finding}")
        if any(not finding.ok for finding in findings):
            failed = True

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
