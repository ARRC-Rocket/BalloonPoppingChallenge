"""The shared comparison, tested for what it lets through.

A golden master is only as good as the thing doing the comparing, and this
helper is the only place both scenarios pass through. Every test here is a way
of being wrong that a real regression could produce: a diverged point, a
baseline that lost a coordinate, a truncated run, a tail the baseline cannot
see.

No simulation. These run in milliseconds and need nothing installed beyond
numpy, so they can stay in the fast suite while the scenarios themselves sit
behind the slow gate.
"""

import unittest

import numpy as np

from tests.position_tolerance import (
    CONTINUITY_FACTOR,
    POSITION_VECTOR_ATOL,
    assert_positions_match,
)


def _straight_line(rows=20, step=25.0):
    """A trajectory shaped like the real ones: moving, and by a known amount.

    Twenty-five metres per sample is the median the shipped scenario-0 baseline
    actually shows, so the continuity bound derived from it here is the same
    order as the one derived in a real run.
    """
    return np.arange(rows, dtype=float)[:, None] * np.array([step, 0.0, 0.0])


class TestTheComparison(unittest.TestCase):
    def compare(self, actual, expected, row_count_abs_tol=1):
        assert_positions_match(self, actual, expected, "rocket", row_count_abs_tol)

    def test_identical_trajectories_pass(self):
        baseline = _straight_line()

        self.compare(baseline.copy(), baseline)

    def test_a_diverged_point_fails(self):
        baseline = _straight_line()
        actual = baseline.copy()
        actual[10] += [POSITION_VECTOR_ATOL * 2, 0.0, 0.0]

        with self.assertRaisesRegex(AssertionError, "3D position error"):
            self.compare(actual, baseline)

    def test_error_split_across_axes_still_fails(self):
        """The reason the bound is on the vector rather than on each axis.

        Three axes at 0.34 m each are inside a half-metre per-axis floor while
        the point has moved 0.59 m.
        """
        baseline = _straight_line()
        actual = baseline.copy()
        per_axis = POSITION_VECTOR_ATOL * 0.68
        actual[5] += [per_axis, per_axis, per_axis]

        self.assertLess(per_axis, POSITION_VECTOR_ATOL, "the premise")
        with self.assertRaisesRegex(AssertionError, "3D position error"):
            self.compare(actual, baseline)


class TestTheTailTheBaselineCannotCheck(unittest.TestCase):
    """The row the count tolerance allows, which used to be discarded.

    A couple of full steps of difference can cross a downsampling boundary and
    produce one extra sample. It has no counterpart in the baseline, so the
    comparison never sees it. Measured before this: five correct rows plus a
    sixth reading (1e9, -1e9, 1e9) passed.
    """

    def compare(self, actual, expected):
        assert_positions_match(self, actual, expected, "rocket", 1)

    def test_an_absurd_extra_row_fails(self):
        baseline = _straight_line()
        actual = np.vstack([baseline, [1e9, -1e9, 1e9]])

        with self.assertRaisesRegex(AssertionError, "not a continuation"):
            self.compare(actual, baseline)

    def test_an_extra_row_that_continues_the_trajectory_passes(self):
        """The other half, or the fix could be "reject every extra row".

        The row-count tolerance exists for a reason and this must not remove it.
        """
        baseline = _straight_line()
        continuation = baseline[-1] + (baseline[-1] - baseline[-2])
        actual = np.vstack([baseline, continuation])

        self.compare(actual, baseline)

    def test_the_bound_comes_from_the_trajectory_rather_than_a_constant(self):
        """Two trajectories at different speeds get different allowances.

        A fixed metre value would either reject the fast one's continuation or
        wave the slow one's nonsense through. Same displacement, opposite
        verdicts.
        """
        jump = np.array([200.0, 0.0, 0.0])

        fast = _straight_line(step=100.0)
        self.compare(np.vstack([fast, fast[-1] + jump]), fast)

        slow = _straight_line(step=1.0)
        with self.assertRaisesRegex(AssertionError, "not a continuation"):
            self.compare(np.vstack([slow, slow[-1] + jump]), slow)

    def test_a_stationary_trajectory_does_not_demand_an_identical_tail(self):
        """Zero movement measures nothing, so it falls back to the tolerance.

        Without this, a trajectory that never moves would have an allowance of
        zero and reject any extra row at all.
        """
        baseline = np.zeros((10, 3))

        self.compare(
            np.vstack([baseline, [POSITION_VECTOR_ATOL * 0.5, 0.0, 0.0]]), baseline
        )
        with self.assertRaisesRegex(AssertionError, "not a continuation"):
            self.compare(np.vstack([baseline, [1e9, 0.0, 0.0]]), baseline)

    def test_it_works_on_balloon_shaped_arrays_too(self):
        """(T, N, 3), where the jump is the worst balloon rather than a point."""
        baseline = np.stack([_straight_line(), _straight_line(step=10.0)], axis=1)
        wrong = baseline[-1].copy()
        wrong[1] += [1e9, 0.0, 0.0]

        with self.assertRaisesRegex(AssertionError, "not a continuation"):
            assert_positions_match(
                self, np.concatenate([baseline, wrong[None]]), baseline, "balloon", 1
            )

    def test_the_factor_is_the_thing_being_relied_on(self):
        """Pins what the allowance is, so a change to it is deliberate."""
        self.assertEqual(CONTINUITY_FACTOR, 3.0)


class TestTheSchema(unittest.TestCase):
    """Shapes, before numpy gets a chance to broadcast a wrong one into place."""

    def compare(self, actual, expected):
        assert_positions_match(self, actual, expected, "rocket", 1)

    def test_a_baseline_that_lost_two_coordinates_fails(self):
        with self.assertRaisesRegex(AssertionError, "ending in XYZ"):
            self.compare(np.zeros((5, 3)), np.zeros((5, 1)))

    def test_a_two_coordinate_pair_fails(self):
        """Both sides being (T, 2) is the one that would silently hold.

        The norm runs happily over two axes and the 3D oracle becomes a 2D one,
        which is what a regenerated baseline would then freeze in place.
        """
        with self.assertRaisesRegex(AssertionError, "ending in XYZ"):
            self.compare(np.zeros((5, 2)), np.zeros((5, 2)))

    def test_a_non_finite_value_fails_even_in_the_unmatched_row(self):
        """Checked over the whole array, before anything is sliced off."""
        baseline = _straight_line()
        actual = np.vstack([baseline, [np.nan, 0.0, 0.0]])

        with self.assertRaisesRegex(AssertionError, "non-finite"):
            self.compare(actual, baseline)

    def test_an_empty_trajectory_fails(self):
        with self.assertRaisesRegex(AssertionError, "empty axis"):
            self.compare(np.zeros((0, 3)), np.zeros((0, 3)))

    def test_too_many_missing_rows_fails(self):
        baseline = _straight_line()

        with self.assertRaisesRegex(AssertionError, "row count"):
            self.compare(baseline[:-3], baseline)


if __name__ == "__main__":
    unittest.main()
