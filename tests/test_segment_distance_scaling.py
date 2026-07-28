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

        The branch selected below the tolerance pins s to zero, which is right
        only when the directions really are parallel: then every s gives the
        same distance. For merely close to parallel it is wrong, so widening
        the tolerance moves pairs into a branch that does not answer their
        question.

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

        Worth being straight about what this does and does not pin. It does not
        fail if the tolerance is loosened: measured, 0.5 and 0.05 both pass it.
        The reason is that the parallel branch is not simply ``s = 0``. When the
        ``t`` that follows clamps out of range, ``s`` is recomputed from the
        clamped ``t``, and for most geometries that recovers the right answer.
        The branch only gives a wrong result when ``s = 0`` produces a ``t``
        strictly inside the segment, which is what the millimetre case above
        does and what an angle on its own does not arrange.

        So the tolerance's value is not observable anywhere in a wide range,
        because a pair near enough to parallel for it to matter is one the
        parallel branch answers correctly anyway. What is observable, and what
        the other tests here pin, is relative against absolute.
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


if __name__ == "__main__":
    unittest.main()
