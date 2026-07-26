"""Unit tests for the balloon pop-detection geometry.

Pop detection is the scoring mechanism: a balloon is popped when the swept path
of the rocket passes within ``balloon.radius`` of the balloon over a timestep.
The geometry lives in two pure-numpy helpers on ``BalloonPoppingEnv``:

* ``_segment_distance_squared_batch`` -- squared minimum distance between one
  segment (the rocket sweep) and N segments (the balloon sweeps).
* ``_detect_pops`` -- flips released balloons (status 1) to popped (status 2)
  when that squared distance is within the squared radius.

Both are numerically pure, but importing the env module pulls in the simulation
stack, so these are guarded with ``skipUnless`` like the other runtime tests.
"""

import unittest

try:
    import numpy as np

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

    _STACK_AVAILABLE = True
except ImportError:
    _STACK_AVAILABLE = False


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestSegmentDistanceSquaredBatch(unittest.TestCase):
    """Squared closest-distance between the rocket segment and N balloon segments."""

    def _distance_squared(self, start_a, end_a, starts_b, ends_b):
        return BalloonPoppingEnv._segment_distance_squared_batch(
            start_a, end_a, starts_b, ends_b
        )

    def test_parallel_segments(self):
        # Two parallel unit segments offset by 3 along y -> distance^2 = 9.
        result = self._distance_squared([0, 0, 0], [1, 0, 0], [[0, 3, 0]], [[1, 3, 0]])
        np.testing.assert_allclose(result, [9.0], atol=1e-9)

    def test_intersecting_segments_have_zero_distance(self):
        # A along x, B along y crossing it at (1, 0, 0) -> they touch.
        result = self._distance_squared([0, 0, 0], [2, 0, 0], [[1, -1, 0]], [[1, 1, 0]])
        np.testing.assert_allclose(result, [0.0], atol=1e-9)

    def test_degenerate_segment_a_is_a_point(self):
        # A collapses to the origin; nearest point on B is (0, 5, 0) -> 25.
        result = self._distance_squared([0, 0, 0], [0, 0, 0], [[-1, 5, 0]], [[1, 5, 0]])
        np.testing.assert_allclose(result, [25.0], atol=1e-9)

    def test_degenerate_segment_b_is_a_point(self):
        # B collapses to (0, 4, 0); nearest point on A is the origin -> 16.
        result = self._distance_squared([-1, 0, 0], [1, 0, 0], [[0, 4, 0]], [[0, 4, 0]])
        np.testing.assert_allclose(result, [16.0], atol=1e-9)

    def test_colinear_segments_clamp_to_endpoints(self):
        # Colinear but disjoint along x; closest points are the facing
        # endpoints (1,0,0) and (5,0,0) -> distance^2 = 16.
        result = self._distance_squared([0, 0, 0], [1, 0, 0], [[5, 0, 0]], [[6, 0, 0]])
        np.testing.assert_allclose(result, [16.0], atol=1e-9)

    def test_batch_returns_one_distance_per_segment(self):
        # A is a point at the origin; three point-segments at known ranges.
        starts_b = [[0, 3, 0], [0, 0, 4], [6, 0, 0]]
        result = self._distance_squared([0, 0, 0], [0, 0, 0], starts_b, starts_b)
        self.assertEqual(result.shape, (3,))
        np.testing.assert_allclose(result, [9.0, 16.0, 36.0], atol=1e-9)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestDetectPops(unittest.TestCase):
    """``_detect_pops`` flips only released balloons within radius to popped."""

    # Status codes used throughout: 0 = on ground, 1 = released, 2 = popped.
    GROUND, RELEASED, POPPED = 0, 1, 2

    @staticmethod
    def _env_with_state(balloon_positions, balloon_status, rocket_now, radius):
        """Build a bare env exercising ``_detect_pops`` in isolation.

        ``__init__`` is bypassed on purpose: it only builds the Gym action and
        observation spaces, which are irrelevant to the pop geometry. The method
        under test reads just these four attributes.
        """
        env = BalloonPoppingEnv.__new__(BalloonPoppingEnv)
        env.balloon_parameters = {"radius": radius}
        positions = np.asarray(balloon_positions, dtype=float)
        count = positions.shape[0]
        env._balloon_states = np.zeros((count, 6))
        env._balloon_states[:, :3] = positions
        env._balloon_status = np.asarray(balloon_status, dtype=int).reshape(count, 1)
        env._rocket_states = np.full(13, np.nan)
        env._rocket_states[:3] = np.asarray(rocket_now, dtype=float)
        return env

    def _run(self, positions, status, radius=2.0):
        """Sweep the rocket from (-5,0,0) to (5,0,0) over static balloons."""
        positions = np.asarray(positions, dtype=float)
        env = self._env_with_state(
            positions, status, rocket_now=[5, 0, 0], radius=radius
        )
        env._detect_pops(
            previous_balloon_positions=positions,
            previous_rocket_position=np.array([-5.0, 0.0, 0.0]),
        )
        return env._balloon_status[:, 0]

    def test_released_balloon_in_path_is_popped(self):
        status = self._run([[0, 0, 0]], [self.RELEASED])
        np.testing.assert_array_equal(status, [self.POPPED])

    def test_distant_released_balloon_is_not_popped(self):
        # 5 m off the rocket's x-axis sweep, radius 2 -> miss.
        status = self._run([[0, 5, 0]], [self.RELEASED])
        np.testing.assert_array_equal(status, [self.RELEASED])

    def test_grounded_balloon_is_not_popped_even_in_path(self):
        # Geometrically a hit, but a ground balloon (status 0) is never checked.
        status = self._run([[0, 0, 0]], [self.GROUND])
        np.testing.assert_array_equal(status, [self.GROUND])

    def test_already_popped_balloon_stays_popped(self):
        status = self._run([[0, 0, 0]], [self.POPPED])
        np.testing.assert_array_equal(status, [self.POPPED])

    def test_distance_exactly_equal_to_radius_is_popped(self):
        # Distance 2 with radius 2: the radius bound is inclusive (<=).
        status = self._run([[0, 2, 0]], [self.RELEASED], radius=2.0)
        np.testing.assert_array_equal(status, [self.POPPED])

    def test_no_released_balloons_is_a_noop(self):
        # Only ground/popped balloons -> early return, nothing changes.
        status = self._run([[0, 0, 0], [0, 0, 0]], [self.GROUND, self.POPPED])
        np.testing.assert_array_equal(status, [self.GROUND, self.POPPED])

    def test_mixed_states_pop_only_the_correct_indices(self):
        # Hit+released, miss+released, hit+ground -> only the first pops.
        positions = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        status = self._run(positions, [self.RELEASED, self.RELEASED, self.GROUND])
        np.testing.assert_array_equal(status, [self.POPPED, self.RELEASED, self.GROUND])

    def _run_moving(
        self, prev_balloon, cur_balloon, status, prev_rocket, cur_rocket, radius=2.0
    ):
        """Both the rocket and the balloons sweep a segment over the timestep.

        Unlike ``_run`` (static balloons), the balloon's previous and current
        positions differ, so pop detection must use the swept balloon segment --
        the scenario-#1 case where balloons drift on the wind.
        """
        env = self._env_with_state(
            cur_balloon, status, rocket_now=cur_rocket, radius=radius
        )
        env._detect_pops(
            previous_balloon_positions=np.asarray(prev_balloon, dtype=float),
            previous_rocket_position=np.asarray(prev_rocket, dtype=float),
        )
        return env._balloon_status[:, 0]

    def test_moving_balloon_crossing_rocket_path_is_popped(self):
        # Rocket sweeps along +x through (5,0,0); the balloon sweeps across the
        # x-axis at x=5 in the same step, so the swept paths intersect -> pop.
        status = self._run_moving(
            prev_balloon=[[5, 5, 0]],
            cur_balloon=[[5, -5, 0]],
            status=[self.RELEASED],
            prev_rocket=[0, 0, 0],
            cur_rocket=[10, 0, 0],
        )
        np.testing.assert_array_equal(status, [self.POPPED])

    def test_moving_balloon_swept_closest_approach_within_radius_is_popped(self):
        # The swept paths never cross, but the balloon's sweep passes 1.5 m from
        # the rocket's sweep (radius 2) -> pop, exercising the swept distance.
        status = self._run_moving(
            prev_balloon=[[5, 3, 0]],
            cur_balloon=[[5, 1.5, 0]],
            status=[self.RELEASED],
            prev_rocket=[0, 0, 0],
            cur_rocket=[10, 0, 0],
        )
        np.testing.assert_array_equal(status, [self.POPPED])

    def test_moving_balloon_passing_outside_radius_is_not_popped(self):
        # The balloon sweeps but stays 3 m from the rocket's path (radius 2).
        status = self._run_moving(
            prev_balloon=[[5, 5, 0]],
            cur_balloon=[[5, 3, 0]],
            status=[self.RELEASED],
            prev_rocket=[0, 0, 0],
            cur_rocket=[10, 0, 0],
        )
        np.testing.assert_array_equal(status, [self.RELEASED])


if __name__ == "__main__":
    unittest.main()
