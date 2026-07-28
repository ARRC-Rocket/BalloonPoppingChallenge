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
# One metre is sized from the measured drift instead. Two runs on one machine
# are bit identical, and regenerating on different hardware moves the last few
# significant digits, well under a millimetre. TestTheCapStaysBelowTheRadius
# keeps this honest against the scenario's configured radius. Raising it past
# a metre would bring back the case the per-coordinate floor used to cover.
POSITION_VECTOR_ATOL = 1.0


def assert_positions_match(test_case, actual, expected, label, row_count_abs_tol):
    """Compare a downsampled trajectory against its baseline.

    The bound is on the displacement, not on each axis. Three axes each 0.99 m
    off pass a per-axis floor of a metre while the point has moved 1.71 m, which
    is more than the balloon radius the score depends on.

    The row count is checked first, so a truncated trajectory cannot pass
    vacuously on its shorter prefix, and the comparison then runs over the
    overlap. Subtracting the full arrays would raise a broadcasting ValueError
    on exactly the one-row difference the count tolerance exists to allow.

    A NaN anywhere in ``actual`` gives a NaN norm, and NaN fails the comparison
    rather than passing it.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    test_case.assertLessEqual(
        abs(len(actual) - len(expected)),
        row_count_abs_tol,
        f"{label} downsampled row count {len(actual)} drifted from baseline "
        f"{len(expected)} by more than {row_count_abs_tol} rows",
    )

    overlap = min(len(expected), len(actual))
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
