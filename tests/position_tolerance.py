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
