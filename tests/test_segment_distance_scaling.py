"""The pop geometry has to decide by angle, not by how long the segments are.

``_segment_distance_squared_batch`` treats two directions as parallel when
``a * e - b * b`` is small. That quantity is in m**4 and depends on the segment
lengths as much as on the angle, so comparing it against one absolute number
made the decision move with scale: shrink two perpendicular segments far enough
and they were classified as parallel, which takes a branch that pins ``s`` to
zero and answers a different question.

A separate file from ``test_pop_detection``, which covers which branch each case
reaches. This covers the property that should hold whatever the branching looks
like, so a later rewrite of the classification does not have to edit it.
"""

import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

# Scenario 1's balloon radius. The failing case below is built around it because
# the point is that the error crosses it.
BALLOON_RADIUS = 1.5


def _brute_force_distance(start_a, end_a, start_b, end_b, samples=4001):
    """Sampled closest approach, as an oracle the implementation cannot bias.

    Dense enough that its own discretisation error is far below anything
    asserted here, and it knows nothing about parallel or degenerate cases.
    """
    fractions = np.linspace(0.0, 1.0, samples)[:, None]
    points_a = start_a + fractions * (end_a - start_a)
    points_b = start_b + fractions * (end_b - start_b)
    # In chunks. The whole difference tensor at 4001 samples is
    # 4001 * 4001 * 3 * 8 bytes, about 366 MiB, and squaring it can hold two of
    # those at once. That is a lot of memory for one assertion, and this file
    # makes the call eight times.
    best = np.inf
    for start in range(0, len(points_a), 256):
        block = points_a[start : start + 256]
        best = min(
            best,
            float(
                np.sqrt(((block[:, None, :] - points_b[None, :, :]) ** 2).sum(-1)).min()
            ),
        )
    return best


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheParallelTestDoesNotDependOnScale(unittest.TestCase):
    def distance(self, start_a, end_a, start_b, end_b):
        squared = BalloonPoppingEnv._segment_distance_squared_batch(
            np.asarray(start_a, dtype=float),
            np.asarray(end_a, dtype=float),
            np.asarray([start_b], dtype=float),
            np.asarray([end_b], dtype=float),
        )
        return float(np.sqrt(squared[0]))

    def test_two_short_perpendicular_segments_are_not_parallel(self):
        """The case the absolute tolerance got wrong, and it changes a score.

        Both segments are 1 mm long and meet at exactly ninety degrees, so
        ``a = e = 1e-6`` and ``b = 0``, which puts the denominator at exactly
        1e-12. Compared against an absolute 1e-12 with ``>``, that is not
        greater, so the two were called parallel. The branch that follows pins
        ``s`` to zero and answered 1.5005 m, while the segments really pass
        within 1.4995 m: a pop reported as a miss.
        """
        start_a, end_a = [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]
        start_b, end_b = [1.5005, -0.0005, 0.0], [1.5005, 0.0005, 0.0]

        # The premise, so this cannot quietly become a near-parallel case.
        direction_a = np.asarray(end_a) - np.asarray(start_a)
        direction_b = np.asarray(end_b) - np.asarray(start_b)
        cosine = abs(direction_a @ direction_b) / (
            np.linalg.norm(direction_a) * np.linalg.norm(direction_b)
        )
        self.assertAlmostEqual(cosine, 0.0, places=12, msg="not perpendicular")

        measured = self.distance(start_a, end_a, start_b, end_b)
        truth = _brute_force_distance(
            np.asarray(start_a),
            np.asarray(end_a),
            np.asarray(start_b),
            np.asarray(end_b),
        )

        self.assertAlmostEqual(measured, truth, places=6)
        self.assertLessEqual(
            measured, BALLOON_RADIUS, "this is a pop and was reported as a miss"
        )

    def test_a_nearly_parallel_pair_still_gets_the_real_closest_approach(self):
        """The tolerance is observable, and the first version of this got it wrong.

        Nothing is skipped now, the stationary point is always computed. The
        case is kept for showing an absolute parallel test was scale dependent.

        At a chosen 1e-12 this pair came back as 1.5000004 m where the true
        closest approach is 1.4999995 m, which against a 1.5 m radius is a pop
        reported as a miss. That is the same defect this file exists to fix,
        moved rather than removed. The tolerance is now derived from double
        precision instead.
        """
        start_a, end_a = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        start_b = np.array([-1.0, 1.5000013, 0.0])
        end_b = np.array([1.0, 1.4999995, 0.0])

        # The premise: close enough to parallel that a loose tolerance catches
        # it, and not parallel.
        direction_a, direction_b = end_a - start_a, end_b - start_b
        a_coeff = float(direction_a @ direction_a)
        e_coeff = float(direction_b @ direction_b)
        b_coeff = float(direction_a @ direction_b)
        relative = abs(a_coeff * e_coeff - b_coeff * b_coeff) / (a_coeff * e_coeff)
        self.assertLess(relative, 1e-12, "not near enough to parallel")
        self.assertGreater(relative, 0.0, "exactly parallel proves nothing here")

        measured = self.distance(start_a, end_a, start_b, end_b)
        truth = _brute_force_distance(start_a, end_a, start_b, end_b)

        self.assertAlmostEqual(measured, truth, places=9)
        self.assertLessEqual(
            measured, BALLOON_RADIUS, "this is a pop and was reported as a miss"
        )

    def test_the_same_geometry_answers_the_same_at_every_scale(self):
        """Distance is homogeneous, so the answer over the scale must be flat.

        Nothing here is near parallel or degenerate at any scale; the only
        thing changing is the size. An answer that moves is the classification
        moving, which is what an absolute tolerance on an m**4 quantity does.
        """
        shape_a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        shape_b = (np.array([0.5, -0.5, 0.25]), np.array([0.5, 0.5, 0.25]))

        ratios = []
        for scale in (1e-4, 1e-3, 1e-2, 1.0, 1e2, 1e3):
            with self.subTest(scale=scale):
                measured = self.distance(
                    shape_a[0] * scale,
                    shape_a[1] * scale,
                    shape_b[0] * scale,
                    shape_b[1] * scale,
                )
                truth = _brute_force_distance(
                    shape_a[0] * scale,
                    shape_a[1] * scale,
                    shape_b[0] * scale,
                    shape_b[1] * scale,
                )
                self.assertAlmostEqual(measured / scale, truth / scale, places=9)
                ratios.append(measured / scale)

        self.assertAlmostEqual(min(ratios), max(ratios), places=9)

    def test_a_moderate_angle_is_not_treated_as_parallel(self):
        """The tolerance has to be small, not merely relative.

        Everything else here is either perpendicular or exactly parallel, and
        the denominator is ``a * e * sin(angle)**2``, so a perpendicular case
        survives any relative tolerance below one. Measured: setting the
        tolerance to 0.5, which calls anything within forty-five degrees
        parallel, passed every other test in this file.

        What this pins is the ordinary case, a clear angle where the answer has
        to match the oracle. The delicate ones are in TestANearParallelInteriorMinimum.
        """
        angle = np.radians(20.0)
        start_a, end_a = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        direction_b = np.array([np.cos(angle), np.sin(angle), 0.0])
        start_b = np.array([0.7, 0.3, 0.2])
        end_b = start_b + 0.6 * direction_b

        self.assertAlmostEqual(np.sin(angle) ** 2, 0.117, places=3)

        measured = self.distance(start_a, end_a, start_b, end_b)
        truth = _brute_force_distance(start_a, end_a, start_b, end_b)

        self.assertAlmostEqual(measured, truth, places=6)

    def test_genuinely_parallel_segments_still_take_the_parallel_answer(self):
        """The other side, or the fix could be "never call anything parallel".

        Two parallel segments have no unique closest pair, and the answer is
        the separation between the lines. Checked at two scales, because the
        relative test has to keep recognising them when they shrink as well.
        """
        for scale in (1e-3, 1.0):
            with self.subTest(scale=scale):
                measured = self.distance(
                    [0.0, 0.0, 0.0],
                    [scale, 0.0, 0.0],
                    [0.0, 2.0 * scale, 0.0],
                    [scale, 2.0 * scale, 0.0],
                )

                self.assertAlmostEqual(measured, 2.0 * scale, places=9)

    def test_a_truly_degenerate_segment_is_still_a_point(self):
        """Zero length is a different case from short, and stays one."""
        measured = self.distance(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 0.0]
        )

        self.assertAlmostEqual(measured, 5.0, places=12)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestANearParallelInteriorMinimum(unittest.TestCase):
    """The case four edges do not contain.

    The edges hold the minimum only for exactly parallel directions. Merely
    near parallel, the stationary point can lie strictly inside both segments.
    """

    RADIUS = 1.5

    # Away from the midpoints on purpose. At s = t = 0.5 the two products in
    # the numerators come out equal in magnitude but not in value, so a
    # symmetric fixture misses the cancellation.
    S_TRUE = 0.35
    T_TRUE = 0.70

    def _pair(self):
        angle = np.sqrt(10 * np.finfo(float).eps)
        length = 20.0
        separation = self.RADIUS - 1e-14
        direction_a = np.array([length, 0.0, 0.0])
        direction_b = length * np.array([np.cos(angle), np.sin(angle), 0.0])
        start_a = np.zeros(3)
        start_b = (
            self.S_TRUE * direction_a
            - self.T_TRUE * direction_b
            - np.array([0.0, 0.0, separation])
        )
        return start_a, start_a + direction_a, start_b, start_b + direction_b

    def _orderings(self):
        start_a, end_a, start_b, end_b = self._pair()
        return {
            "as written": (start_a, end_a, start_b, end_b),
            "rocket reversed": (end_a, start_a, start_b, end_b),
            "balloon reversed": (start_a, end_a, end_b, start_b),
            "both reversed": (end_a, start_a, end_b, start_b),
            "swapped": (start_b, end_b, start_a, end_a),
        }

    @staticmethod
    def _relative_determinant(start_a, end_a, start_b, end_b):
        direction_a, direction_b = end_a - start_a, end_b - start_b
        a = float(direction_a @ direction_a)
        e = float(direction_b @ direction_b)
        b = float(direction_a @ direction_b)
        return (a * e - b * b) / (a * e)

    def _skipped_interior_pair(self):
        """The pair the removed cutoff would have refused to solve.

        Below 8 * eps, so restoring the cutoff skips its stationary point. The
        asymmetric pair sits above it and cannot catch that regression.
        """
        angle = np.sqrt(7 * np.finfo(float).eps)
        just_inside = np.nextafter(self.RADIUS, 0.0)
        half = 5.0
        return (
            np.array([-half, 0.0, 0.0]),
            np.array([half, 0.0, 0.0]),
            np.array([-half, -half * angle, just_inside]),
            np.array([half, half * angle, just_inside]),
        )

    def test_the_skipped_interior_pair_is_below_the_old_cutoff(self):
        """Both halves, or the fixture is not the case it is named for."""
        relative = self._relative_determinant(*self._skipped_interior_pair())

        self.assertGreater(relative, 0.0, "this pair is exactly parallel")
        self.assertLessEqual(relative, 8 * np.finfo(float).eps)

    def test_the_pair_below_the_old_cutoff_is_still_a_pop(self):
        start_a, end_a, start_b, end_b = self._skipped_interior_pair()

        measured = float(
            np.sqrt(
                BalloonPoppingEnv._segment_distance_squared_batch(
                    start_a, end_a, np.array([start_b]), np.array([end_b])
                )[0]
            )
        )

        self.assertLessEqual(measured, self.RADIUS)

    def test_the_cancelling_pair_is_near_parallel_but_not_parallel(self):
        relative = self._relative_determinant(*self._pair())

        self.assertGreater(relative, 0.0, "this pair is exactly parallel")
        self.assertLess(relative, 1e-13, "this pair is not near parallel enough")

    def test_the_closest_points_are_where_the_fixture_puts_them(self):
        start_a, end_a, start_b, end_b = self._pair()
        closest_a = start_a + self.S_TRUE * (end_a - start_a)
        closest_b = start_b + self.T_TRUE * (end_b - start_b)

        distance = float(np.linalg.norm(closest_a - closest_b))

        self.assertLessEqual(distance, self.RADIUS, "the fixture is not a pop")
        # Well inside, not one representable float inside, so a verdict that
        # flips here is a real loss of digits rather than the last bit.
        self.assertGreater(
            (self.RADIUS - distance) / np.spacing(self.RADIUS),
            10.0,
            "the fixture sits too close to the radius to mean anything",
        )

    def test_the_interior_minimum_is_not_skipped_in_any_ordering(self):
        """Every way of writing the same two segments down.

        A skipped stationary point and cancelling numerators share a symptom.
        Reversing the rocket sweep moves ``b * f - c * e`` across the radius.
        """
        measured = {
            name: float(
                np.sqrt(
                    BalloonPoppingEnv._segment_distance_squared_batch(
                        start_a, end_a, np.array([start_b]), np.array([end_b])
                    )[0]
                )
            )
            for name, (start_a, end_a, start_b, end_b) in self._orderings().items()
        }

        for name, distance in measured.items():
            with self.subTest(ordering=name):
                self.assertLessEqual(
                    distance,
                    self.RADIUS,
                    f"{name} reports {distance!r} against a {self.RADIUS} m "
                    "radius, so a pop is scored as a miss",
                )
        self.assertLessEqual(
            max(measured.values()) - min(measured.values()),
            1e-12,
            f"the orderings disagree: {measured}",
        )


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestEveryCandidateEarnsItsPlace(unittest.TestCase):
    """Deleting an edge from the five candidates changed no other test.

    Both golden masters included. Dropping the ``s = 1`` edge moves the pair
    below from 1.49990 m to 4.05960 m, a pop lost against a 1.5 m radius.
    """

    # From an exhaustive search for a geometry where the s = 1 edge is the only
    # candidate holding the minimum, at scenario magnitudes.
    ROCKET = np.array([[0.0, 0.0, 0.0], [2.248642, 7.195445, 26.380041]])
    BALLOON = np.array(
        [[-2.732624, 5.051852, 16.682862], [3.586303, 10.1662, 32.856179]]
    )

    def test_the_s_edge_holds_the_minimum_here(self):
        truth = _brute_force_distance(*self.ROCKET, *self.BALLOON, samples=20001)

        measured = float(
            np.sqrt(
                BalloonPoppingEnv._segment_distance_squared_batch(
                    self.ROCKET[0],
                    self.ROCKET[1],
                    self.BALLOON[0][None, :],
                    self.BALLOON[1][None, :],
                )[0]
            )
        )

        self.assertAlmostEqual(measured, truth, places=4)
        self.assertLess(measured, 1.5, "this pair has to be a pop for the test to bite")

    def test_no_geometry_in_a_corpus_beats_the_oracle(self):
        """A fixed corpus, so both edge families are covered symmetrically.

        One-sided on purpose: sampling can only overestimate the closest
        approach, so anything missing or wrongly signed lands above the oracle.
        """
        generator = np.random.default_rng(20260729)
        worst = 0.0
        for _ in range(200):
            points = generator.uniform(-30.0, 30.0, size=(4, 3))
            sampled = _brute_force_distance(*points, samples=401)
            measured = float(
                np.sqrt(
                    BalloonPoppingEnv._segment_distance_squared_batch(
                        points[0],
                        points[1],
                        points[2][None, :],
                        points[3][None, :],
                    )[0]
                )
            )
            worst = max(worst, measured - sampled)

        self.assertLessEqual(
            worst, 1e-9, f"exceeded a sampled upper bound by {worst:.3e} m"
        )


if __name__ == "__main__":
    unittest.main()
