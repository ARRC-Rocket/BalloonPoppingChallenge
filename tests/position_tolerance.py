"""Comparing a trajectory against its baseline, shared by both scenarios.

Kept in one place rather than copied into each regression test. The two copies
had already drifted: scenario 1's was covered by its own tests while scenario
0's had a broadcasting bug in the same few lines, and nothing could see it
because each file only tested its own copy.
"""

import numpy as np

# One bound, on the 3D displacement, in absolute terms.
#
# It replaces the per-coordinate max(1 m, 3% of |expected|) rather than joining
# it. That check could never fail on its own: failing it needs some |delta_i| to
# exceed a floor of 1 m, and the displacement is at least |delta_i|, so this cap
# had already failed. Over two million random cases the per-coordinate check
# fired without this one exactly zero times, while this one fired without it a
# million times. Keeping both would have left a branch no test could tell the
# presence of from the absence of.
#
# Deliberately not a fraction of |expected|.
#
# Z in this frame is altitude above sea level, not height above the launch
# site, so |expected| is dominated by where the coordinate origin happens to be
# rather than by anything about the trajectory. A relative bound therefore grows
# with altitude: on the committed scenario 0 baseline, 3% of |expected| exceeds
# the 1.5 m balloon radius at 108 of the 119 sampled points and reaches 48.8 m
# at apogee. That is not a bound on trajectory error, it is a bound on distance
# from sea level.
#
# Half a metre, sized from the measured drift. Two runs on one machine are bit
# identical, and so are runs on numpy 2.4.5/scipy 1.17.1 versus numpy
# 2.5.1/scipy 1.18.0, which is the spread between the lockfile and what CI
# resolves. Cross-machine regeneration moves the last few significant digits.
# So this is several orders of magnitude of headroom either way.
#
# Half rather than a whole metre because scenario 1 applies this cap to the
# rocket and to the balloons separately. Pop detection depends on the distance
# between them, and two independent errors of the full cap in opposite
# directions move that distance by twice the cap. At half a metre the combined
# worst case is 1 m, under the 1.5 m radius; at a whole metre it was 2 m, over
# it. TestTheCapStaysBelowTheRadius asserts the doubled figure.
#
# That comparison is a sanity check on the size of the bound, not a proof that
# a score cannot change. Pop detection sweeps between timesteps while this
# compares downsampled samples, so a bound on sampled positions does not bound
# every swept segment. It says the tolerance is far tighter than the scale at
# which scoring behaviour turns over, which is what it is for.
POSITION_VECTOR_ATOL = 0.5


def assert_positions_match(test_case, actual, expected, label, row_count_abs_tol):
    """Compare a downsampled trajectory against its baseline.

    The bound is on the displacement, not on each axis. Three axes each 0.99 m
    off pass a per-axis floor of a metre while the point has moved 1.71 m, which
    is more than the balloon radius the score depends on.

    Non-finite values are rejected across the *whole* of both arrays, before
    anything is sliced away. Checking after the slice looked equivalent and was
    not: a diverged value in the one extra row the count tolerance allows was
    discarded before it could reach the norm, so the comparison passed on
    exactly the tail a golden master exists to catch.

    The row count is checked first, so a truncated trajectory cannot pass
    vacuously on its shorter prefix, and the comparison then runs over the
    overlap. Subtracting the full arrays would raise a broadcasting ValueError
    on exactly the one-row difference the count tolerance exists to allow.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    # The schema, before anything is subtracted. NumPy broadcasts a trailing
    # axis of length 1 against one of length 3, so a baseline that lost Y and Z
    # compares successfully against a full trajectory. And if both sides become
    # (T, 2), the norm runs happily over two axes and the 3D oracle quietly
    # becomes a 2D one, which is exactly what a regenerated baseline would
    # freeze in place.
    for name, values in ((label, actual), (f"{label} baseline", expected)):
        if values.ndim not in (2, 3) or values.shape[-1] != 3:
            raise AssertionError(
                f"{name} positions must be (T, 3) or (T, N, 3) ending in XYZ, "
                f"got {values.shape}"
            )
        if 0 in values.shape:
            raise AssertionError(f"{name} positions have an empty axis: {values.shape}")

    test_case.assertEqual(
        actual.shape[1:],
        expected.shape[1:],
        f"{label} position shape {actual.shape[1:]} does not match baseline "
        f"{expected.shape[1:]}",
    )

    for name, values in ((label, actual), (f"{label} baseline", expected)):
        finite = np.isfinite(values)
        if not finite.all():
            bad = np.flatnonzero(~finite.reshape(len(values), -1).all(axis=1))
            raise AssertionError(
                f"{name} positions contain a non-finite value, rows "
                f"{bad[:5].tolist()}{'...' if bad.size > 5 else ''}"
            )

    test_case.assertLessEqual(
        abs(len(actual) - len(expected)),
        row_count_abs_tol,
        f"{label} downsampled row count {len(actual)} drifted from baseline "
        f"{len(expected)} by more than {row_count_abs_tol} rows",
    )

    overlap = min(len(expected), len(actual))
    # A tolerated row-count difference must not become an empty comparison.
    test_case.assertGreater(overlap, 0, f"{label} has no overlapping rows")
    unmatched = actual[overlap:]
    actual = actual[:overlap]
    expected = expected[:overlap]

    displacement = np.linalg.norm(actual - expected, axis=-1)
    worst_vector = float(np.max(displacement))
    test_case.assertLessEqual(
        worst_vector,
        POSITION_VECTOR_ATOL,
        f"{label} 3D position error {worst_vector:.4g} m exceeds "
        f"{POSITION_VECTOR_ATOL} m",
    )

    _assert_the_unmatched_tail_is_continuous(test_case, actual, unmatched, label)


# How much further than the largest step seen in the compared rows the one
# tolerated extra row may travel. Three, because the compared rows are a
# measurement of what this trajectory does per sample and the run that produced
# the extra row is the same flight; anything within a few times the observed
# step is a continuation, and anything far outside it is not this trajectory.
CONTINUITY_FACTOR = 3.0


def _assert_the_unmatched_tail_is_continuous(test_case, compared, unmatched, label):
    """The row the baseline has nothing to compare against still has to be real.

    The row-count tolerance exists because a couple of full steps of difference
    can cross a downsampling boundary and produce one extra sample. That row has
    no counterpart in the baseline, so the comparison above never sees it, and
    an earlier version simply discarded it. Measured: five correct rows plus a
    sixth reading (1e9, -1e9, 1e9) passed, because the extra row is finite, the
    count difference is inside tolerance, and the wrong data was sliced away
    before the norm.

    There is nothing to compare it against, so this asks the only thing that can
    be asked of it: that the trajectory is still continuous into it. The bound
    comes from the compared rows themselves rather than a number written here,
    since what one sample of this flight covers is exactly what those rows
    measure. In the shipped baselines the rocket moves at most about 71 m per
    sample and the balloons about 16 m, so the gap between a continuation and
    the case above is seven orders of magnitude.
    """
    if len(unmatched) == 0 or len(compared) < 2:
        return

    steps = np.linalg.norm(np.diff(compared, axis=0), axis=-1)
    largest_step = float(np.max(steps))
    if largest_step == 0.0:
        # A trajectory that never moves says nothing about how far a sample can
        # travel, so fall back to the tolerance rather than demanding the extra
        # row be identical.
        largest_step = POSITION_VECTOR_ATOL
    allowed = CONTINUITY_FACTOR * largest_step

    jump = float(np.max(np.linalg.norm(unmatched[0] - compared[-1], axis=-1)))
    test_case.assertLessEqual(
        jump,
        allowed,
        f"{label} has {len(unmatched)} row(s) the baseline cannot check, and the "
        f"first moves {jump:.4g} m from the last compared row, more than "
        f"{CONTINUITY_FACTOR:g}x the largest step in the comparison "
        f"({largest_step:.4g} m). That is not a continuation of this trajectory",
    )


def launch_step(positions, record_step):
    """The step the rocket first has a state at, on the run's own clock.

    Everything above compares a *launch-relative* trajectory: both scenarios
    slice their positions from the first finite row, so the origin is re-zeroed
    on launch and nothing that follows carries when the flight happened. That is
    the hole this closes. Measured against the committed baselines, moving the
    rocket trajectory later while keeping the total step count passed every
    check in both files up to 60 steps (0.60 s) on scenario 0 and 62 (0.62 s) on
    scenario 1. Neither boundary is a clock: 61 and 63 are simply where the
    downsampled row count finally drifts past ``ROW_COUNT_ABS_TOL``.

    It matters most on scenario 1, which pops nothing and so has no absolute
    anchor for the rocket at all, while the balloons it is scored against run on
    the absolute clock from step 0. The two arrays are on different clocks and
    pop detection depends on their relative phase.

    Scenario 0 is partly covered already, because ``pop_step`` is an absolute
    step index. Displace it consistently, pops moved with the flight, and the
    pop-step comparison rejects a single step, so the largest displacement that
    passes there is 0. That only holds while something pops.

    The step comes out of the record's own time axis rather than off the row
    number, for the same reason scenario 0 reads its pop steps that way:
    ``step()`` increments ``current_step`` before it appends the record, so row
    i is step i+1, and writing that offset down here would make this stop
    meaning what its name says if the logging ever started somewhere else.
    """
    positions = np.asarray(positions, dtype=float)
    record_step = np.asarray(record_step)
    if len(record_step) != len(positions):
        raise AssertionError(
            f"the clock has {len(record_step)} entries and the trajectory has "
            f"{len(positions)}, so a row cannot be dated"
        )
    launched = np.flatnonzero(np.isfinite(positions).all(axis=1))
    if launched.size == 0:
        raise AssertionError("rocket never produced a finite post-launch position")
    return int(record_step[launched[0]])


def assert_launch_step_matches(test_case, actual, expected, step_count_abs_tol):
    """Compare a launch step against its baseline, in whole steps.

    Absolute, and the same tolerance the flight duration gets, for the same
    reason: this is a deterministic run, so the only drift to absorb is a step
    or two of cross-platform jitter in when the integrator first reports a
    state. A percentage would scale with how late the launch happens to be.
    """
    test_case.assertLessEqual(
        abs(actual - expected),
        step_count_abs_tol,
        f"the rocket launched at step {actual}, and the baseline says "
        f"{expected}. The trajectory comparison cannot see this: it is "
        f"re-zeroed on the first flown row",
    )


def displace_flight_in_time(positions, steps):
    """The same flight, launched ``steps`` later, over the same step count.

    The attack the anchor above exists to reject, built here so both scenarios
    run the identical one. The rocket sits unlaunched for ``steps`` more steps
    and the tail is dropped, so the episode length, the trajectory shape and
    every sampled value are unchanged; only the date on them moves.
    """
    positions = np.asarray(positions, dtype=float)
    if not 0 < steps < len(positions):
        raise AssertionError(
            f"a displacement of {steps} steps does not fit inside a "
            f"{len(positions)}-step run"
        )
    unlaunched = np.full((steps,) + positions.shape[1:], np.nan)
    return np.concatenate([unlaunched, positions[:-steps]])
