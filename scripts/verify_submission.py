#!/usr/bin/env python3
"""Check a submission's balloon trajectories against what its seed produces.

The balloons are not something a competitor controls. Their flights come out of
the scenario parameters and the seed, and an agent only ever commands the
rocket, so regenerating them gives an independent copy of what the run should
have recorded. Anything that does not match was not produced by the simulator.

Regenerated from the scenario **this repository ships**, not from the copy
inside the submission. A submission carries its own ``scenario_parameters``, and
using those as the oracle would let the file being checked supply the answer it
is checked against: change the rocket inertia and the balloons together, and
everything agrees with everything. The shipped scenario is compared against the
submitted one as its own check, so a run against modified physics is reported
rather than quietly blessed.

Run it against a submission downloaded from the leaderboard::

    uv run python scripts/verify_submission.py 20260728T115315.000Z_team_<id>_submission.json

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

    Submissions are JSON. Reading by suffix keeps the pickle path working for a
    file packed before the change, without a second copy of everything below.

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


def _load_canonical_scenario(scenario_number):
    """The scenario as this repository ships it, which is the actual oracle.

    Returns ``None`` for a number this repository has no scenario for, which is
    itself worth reporting: a submission naming scenario 7 was not produced by
    anything here.
    """
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

    try:
        canonical, _ = load_scenario_parameters(scenario_number)
    except (FileNotFoundError, OSError, TypeError):
        return None
    return canonical


def _parameter_differences(submitted, canonical, path=""):
    """Every leaf where the two parameter trees disagree, as (path, got, want).

    Compared strictly. Measured before relying on it: the environment does not
    write to the parameters it is handed, and a fresh ``load_scenario_parameters``
    equals the dict a finished run carries, so an honest submission produces an
    empty list rather than a pile of floating point noise.
    """
    if isinstance(canonical, dict):
        if not isinstance(submitted, dict):
            return [(path or "<root>", submitted, canonical)]
        differences = []
        for key in sorted(set(canonical) | set(submitted), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in submitted:
                differences.append((child, "<missing>", canonical[key]))
            elif key not in canonical:
                differences.append((child, submitted[key], "<not in the scenario>"))
            else:
                differences += _parameter_differences(
                    submitted[key], canonical[key], child
                )
        return differences

    if isinstance(canonical, (list, tuple)):
        if not isinstance(submitted, (list, tuple)) or len(submitted) != len(canonical):
            return [(path or "<root>", submitted, canonical)]
        differences = []
        for index, (mine, theirs) in enumerate(zip(submitted, canonical)):
            differences += _parameter_differences(mine, theirs, f"{path}[{index}]")
        return differences

    if submitted != canonical:
        return [(path or "<root>", submitted, canonical)]
    return []


def check_scenario_is_official(submission):
    """Establish the oracle before anything is compared against it.

    Returns the findings and the canonical parameters, or ``None`` for them when
    there is no scenario to check against, in which case nothing that needs
    physics can run.

    The scenario number is taken from the submitted parameters and cross checked
    against the one in ``leaderboard_info``. Both are competitor controlled, so
    neither is trusted: naming a different scenario to the one the data came
    from makes the comparison below fail rather than pass.
    """
    world = submission["balloon_world_data"]
    submitted = world["scenario_parameters"]

    if not isinstance(submitted, dict) or not isinstance(
        submitted.get("scenario"), dict
    ):
        return [
            Finding("scenario", False, "the submission carries no scenario section")
        ], None

    number = submitted["scenario"].get("number")
    claimed = submission.get("leaderboard_info", {}).get("scenario_number")
    findings = []
    if claimed is not None and claimed != number:
        findings.append(
            Finding(
                "scenario number",
                False,
                f"the parameters say scenario {number}, leaderboard_info says "
                f"{claimed}",
            )
        )

    canonical = _load_canonical_scenario(number)
    if canonical is None:
        findings.append(
            Finding(
                "scenario",
                False,
                f"this repository ships no scenario {number!r}, so there is "
                "nothing to check the run against",
            )
        )
        return findings, None

    differences = _parameter_differences(submitted, canonical)
    findings.append(
        Finding(
            "scenario parameters are the shipped ones",
            not differences,
            f"identical to scenario {number} as shipped"
            if not differences
            else f"{len(differences)} parameters differ from scenario {number}",
        )
    )
    for where, got, want in differences[:5]:
        findings.append(
            Finding(
                f"parameter {where}",
                False,
                f"submission has {got!r}, the scenario has {want!r}",
            )
        )
    return findings, canonical


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


def check_balloon_trajectories(submission, tolerance, trusted_positions, canonical):
    """The main check: every recorded balloon state came out of the simulator.

    ``canonical`` is the scenario as shipped, never the copy in the submission.
    Regenerating from the submitted parameters would let the file supply its own
    oracle: modify the physics and the balloons together and they agree.

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

    flights, release_at_step = _regenerate_balloon_flights(canonical)

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


def check_claimed_pops_are_reachable(submission, expected_positions, canonical):
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
    decision. A boundary that is one step out moves the distance by about a
    metre; a fabricated pop is out by hundreds. Where it does follow the
    environment it errs towards letting a pop through, because the cost of being
    subtly wrong here is accusing an honest competitor.

    Two of those boundaries used to be missed, and both could only ever have
    produced a false accusation:

    The environment sweeps the first interval from the launch pad, which is
    ``(0, 0, elevation)``, to the first stepped position. The row before that
    one is all NaN, so an interval needing two real endpoints skipped it. The
    pad is in the scenario, so it is reconstructed exactly rather than
    approximated.

    A balloon released on step k is marked released before pops are detected on
    step k, so the environment can pop it over the interval that ends there.
    Reading the status at the start of that interval says it was still on the
    ground, and skipped it.
    """
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

    world = submission["balloon_world_data"]
    records = world["trajectories"]
    radius = float(canonical["balloon"]["radius"])

    status = np.asarray([record["balloon_status"] for record in records], dtype=int)
    claimed_popped = np.flatnonzero((status == 2).any(axis=0))
    if claimed_popped.size == 0:
        return [Finding("claimed pops", True, "the submission claims no pops")]

    rocket = np.asarray(
        [record["rocket_states"][:3] for record in records], dtype=float
    )
    flyable = np.isfinite(rocket).all(axis=1)
    # Put the launch pad back where the environment starts sweeping from, so the
    # first flown interval is checked instead of dropped for having a NaN end.
    # Overwriting the row before it makes it an ordinary interval below.
    if flyable.any():
        first_flown = int(np.argmax(flyable))
        if first_flown >= 1:
            rocket[first_flown - 1] = (
                0.0,
                0.0,
                float(canonical["environment"]["elevation"]),
            )
            flyable[first_flown - 1] = True
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
        # The status at the *end* of the interval. A balloon released on this
        # step is released before the environment detects pops on it, so reading
        # the start would skip the interval it was actually popped over.
        released = status[step + 1, claimed_popped] >= 1
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

    # Not "2 never follows 0". That reads like the rule and is not one: step()
    # marks a balloon released, detects pops, and only then writes the record,
    # so a balloon released and popped on the same step legitimately records 0
    # then 2 with nothing in between. Asserting that transition away would
    # accuse an honest competitor of the one thing they cannot argue with.
    #
    # The rule that does hold is about when, not about the shape of the
    # sequence: no balloon is popped before the step it is released on. Row k
    # holds step k+1, the same offset the trajectory comparison uses.
    release_at_step = np.asarray(world["balloon_release_at_step"], dtype=int)
    popped_rows = status == 2
    early = []
    if release_at_step.shape[0] == status.shape[1]:
        has_popped = popped_rows.any(axis=0)
        first_pop_step = popped_rows.argmax(axis=0) + _TRAJECTORY_OFFSET
        early = np.flatnonzero(has_popped & (first_pop_step < release_at_step))
    findings.append(
        Finding(
            "no balloon is popped before release",
            len(early) == 0,
            "every pop is on or after the balloon's release step"
            if len(early) == 0
            else f"{len(early)} balloons are popped before they are released, "
            f"first {early[:5].tolist()}",
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

    # First, because everything below is measured against it. A submission that
    # names a scenario this repository does not ship, or whose parameters are
    # not the shipped ones, is reported here rather than checked against itself.
    findings, canonical = check_scenario_is_official(submission)
    # These need no physics, so they run whatever the scenario turned out to be.
    findings += check_internal_consistency(submission)
    if canonical is None:
        return findings

    trusted_positions = []
    findings += check_balloon_trajectories(
        submission, tolerance, trusted_positions, canonical
    )
    # Only worth asking where the rocket went if the balloons it is compared
    # against are the ones the seed produced.
    if trusted_positions and all(finding.ok for finding in findings):
        findings += check_claimed_pops_are_reachable(
            submission, trusted_positions[0], canonical
        )
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
