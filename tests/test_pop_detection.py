"""Unit tests for the balloon pop-detection geometry.

Pop detection is the scoring mechanism: a balloon is popped when the swept path
of the rocket passes within ``balloon.radius`` of the balloon over a timestep.
The geometry lives in two pure-numpy helpers on ``BalloonPoppingEnv``:

* ``_segment_distance_squared_batch`` -- squared minimum distance between one
  segment (the rocket sweep) and N segments (the balloon sweeps).
* ``_detect_pops`` -- flips released balloons (status 1) to popped (status 2)
  when that squared distance is within the squared radius.

``_segment_distance_squared_batch`` is a pure function; ``_detect_pops`` is not,
since flipping the status is the whole point of it. Importing the env module
pulls in the simulation stack, so these are guarded with ``skipUnless`` like the
other runtime tests.
"""

import unittest
from importlib.util import find_spec

import numpy as np

# Only the simulation stack is optional. Guarding this package's own imports too
# would turn a renamed symbol or a broken module into a silent skip.
_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


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
        np.testing.assert_allclose(result, [9.0], atol=1e-9, rtol=0)

    def test_intersecting_segments_have_zero_distance(self):
        # A along x, B along y crossing it at (1, 0, 0) -> they touch.
        result = self._distance_squared([0, 0, 0], [2, 0, 0], [[1, -1, 0]], [[1, 1, 0]])
        np.testing.assert_allclose(result, [0.0], atol=1e-9, rtol=0)

    def test_degenerate_segment_a_is_a_point(self):
        # A collapses to the origin; nearest point on B is (0, 5, 0) -> 25.
        result = self._distance_squared([0, 0, 0], [0, 0, 0], [[-1, 5, 0]], [[1, 5, 0]])
        np.testing.assert_allclose(result, [25.0], atol=1e-9, rtol=0)

    def test_degenerate_segment_a_clamps_to_each_end_of_b(self):
        """The clip in that branch, which the case above cannot reach.

        Its point projects to the middle of B, so the clip has nothing to do
        and deleting it leaves the answer at 25. Measured: it survived.

        These two project outside B in each direction, where an unclipped
        projection walks along the infinite line to the point itself and
        returns zero.
        """
        beyond_the_end = self._distance_squared(
            [5, 0, 0], [5, 0, 0], [[0, 0, 0]], [[1, 0, 0]]
        )
        np.testing.assert_allclose(beyond_the_end, [16.0], atol=1e-9)

        before_the_start = self._distance_squared(
            [-4, 0, 0], [-4, 0, 0], [[0, 0, 0]], [[1, 0, 0]]
        )
        np.testing.assert_allclose(before_the_start, [16.0], atol=1e-9)

    def test_degenerate_segment_b_is_a_point(self):
        # B collapses to (0, 4, 0); nearest point on A is the origin -> 16.
        result = self._distance_squared([-1, 0, 0], [1, 0, 0], [[0, 4, 0]], [[0, 4, 0]])
        np.testing.assert_allclose(result, [16.0], atol=1e-9, rtol=0)

    def test_degenerate_segment_b_clamps_to_each_end_of_a(self):
        """The same gap on the other side, which the review that found the
        first one did not mention. It survived for the same reason: the case
        above puts B's point opposite the middle of A.
        """
        beyond_the_end = self._distance_squared(
            [0, 0, 0], [1, 0, 0], [[5, 0, 0]], [[5, 0, 0]]
        )
        np.testing.assert_allclose(beyond_the_end, [16.0], atol=1e-9)

        before_the_start = self._distance_squared(
            [0, 0, 0], [1, 0, 0], [[-4, 0, 0]], [[-4, 0, 0]]
        )
        np.testing.assert_allclose(before_the_start, [16.0], atol=1e-9)

    def test_colinear_segments_clamp_to_endpoints(self):
        # Colinear but disjoint along x; closest points are the facing
        # endpoints (1,0,0) and (5,0,0) -> distance^2 = 16.
        result = self._distance_squared([0, 0, 0], [1, 0, 0], [[5, 0, 0]], [[6, 0, 0]])
        np.testing.assert_allclose(result, [16.0], atol=1e-9, rtol=0)

    def test_batch_returns_one_distance_per_segment(self):
        # A is a point at the origin; three point-segments at known ranges.
        starts_b = [[0, 3, 0], [0, 0, 4], [6, 0, 0]]
        result = self._distance_squared([0, 0, 0], [0, 0, 0], starts_b, starts_b)
        self.assertEqual(result.shape, (3,))
        np.testing.assert_allclose(result, [9.0, 16.0, 36.0], atol=1e-9, rtol=0)


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


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheClampingBranches(unittest.TestCase):
    """The endpoint clamping the fixed cases above do not reach.

    When the closest point on a balloon sweep falls outside its endpoints, the
    parameter is clamped and the rocket-side parameter is recomputed against that
    endpoint. Swapping the two recomputations leaves every other test in this
    file green, so each clamp needs its own case (see #63 discussion).

    Every case here shares one rocket segment, ``SEGMENT_A``. That is what makes
    the mixed batch at the end a real union of the individual cases rather than a
    third unrelated input that happens to look similar. An earlier version used a
    different rocket segment for the batch, and it reached neither the parallel
    nor the high-clamp branch it claimed to cover.
    """

    # One rocket sweep for every case below.
    SEGMENT_A = (np.zeros(3), np.array([-4.5, -4.7, -4.4]))

    # Chosen by searching for pairs where the far-end and near-end
    # recomputations disagree by a wide margin, rather than by picking something
    # that merely reaches the branch: a first attempt reached the branch and
    # still clamped both to the same value, so the mutation survived.
    CLAMPED_HIGH = ([-5.002, 3.969, 3.068], [-0.857, 0.011, -1.096])
    CLAMPED_LOW = ([-4.004, -1.730, -3.597], [-0.977, 7.680, -4.506])
    # Parallel: the same direction rather than a scaled one, so the denominator
    # is exactly zero and s stays at its initialized zero.
    #
    # It used to be 0.7 times direction_a, which left the denominator at
    # 4.55e-13 against a 1e-12 threshold. That is a margin of 2.2, resting on
    # how the multiplication and subtraction round rather than on the segments
    # being parallel, and #75 has just put this suite on a second Python and a
    # second numpy. Identical directions cancel exactly.
    PARALLEL = (
        [3.0, -1.0, 2.0],
        [3.0 - 4.5, -1.0 - 4.7, 2.0 - 4.4],
    )
    # Degenerate: a balloon that did not move over the step.
    DEGENERATE_B = ([2.0, 3.0, 0.0], [2.0, 3.0, 0.0])
    # The rocket's own closest point falls outside its sweep while the balloon's
    # stays inside, so s is clamped and t is never recomputed.
    S_CLAMPED = ([-4.026, -5.843, 4.051], [-0.605, -2.271, 7.394])
    # Almost degenerate and oblique, which is a different thing: see the test.
    NEAR_DEGENERATE_B = (
        [-7.9985, -2.0956, -2.9027],
        [-7.998499407604158, -2.0955998431503545, -2.9026997331907473],
    )

    def _distance_squared(self, start_a, end_a, starts_b, ends_b):
        return BalloonPoppingEnv._segment_distance_squared_batch(
            start_a, end_a, starts_b, ends_b
        )

    @staticmethod
    def _where_the_minimum_lies(start_a, end_a, start_b, end_b, epsilon=1e-12):
        """Name the region a pair's closest approach falls in.

        This used to name the *branch of the implementation* a pair reached, and
        duplicated its arithmetic to do so. That drifted twice: the threshold
        here stayed absolute while production moved to a relative one, and then
        production stopped having a parallel branch at all, since it now
        evaluates the stationary point and all four edges and takes the smallest.
        Both times this helper went on confidently naming a branch that was no
        longer there.

        So it describes the geometry instead: whether either sweep is a point,
        whether the two directions are parallel, and where the unconstrained
        minimum of the distance falls relative to the two segments. Those are
        facts about the fixture, and they stay true however the minimum is
        computed. What they are for is unchanged: a case is otherwise only
        *believed* to exercise what it was chosen for, which is how the batch
        below came to claim four situations while covering two.
        """
        direction_a = np.asarray(end_a, dtype=float) - np.asarray(start_a, dtype=float)
        direction_b = np.asarray(end_b, dtype=float) - np.asarray(start_b, dtype=float)
        offset = np.asarray(start_a, dtype=float) - np.asarray(start_b, dtype=float)

        a_coeff = float(direction_a @ direction_a)
        if a_coeff <= epsilon:
            return "degenerate_a"
        e_coeff = float(direction_b @ direction_b)
        if e_coeff <= epsilon:
            return "degenerate_b"

        b_coeff = float(direction_a @ direction_b)
        f_coeff = float(direction_b @ offset)
        # Exactly parallel, by the directions rather than by a threshold. This
        # used to copy production's 8 * eps test, which meant it called a pair
        # parallel that is not, and it went stale the moment production stopped
        # having such a test at all.
        normal = np.cross(direction_a, direction_b)
        denominator = float(normal @ normal)
        if denominator == 0.0:
            return "parallel"

        # Same identity production uses, so this describes the geometry rather
        # than reproducing a subtraction that loses digits near parallel.
        s_unclipped = float(np.cross(direction_b, offset) @ normal) / denominator
        s_param = np.clip(s_unclipped, 0.0, 1.0)
        t_param = (b_coeff * s_param + f_coeff) / e_coeff
        if t_param < 0.0:
            return "t_too_low"
        if t_param > 1.0:
            return "t_too_high"
        if not 0.0 <= s_unclipped <= 1.0:
            return "s_clamped"
        return "interior"

    @staticmethod
    def _brute_force(start_a, end_a, start_b, end_b, samples=2001, chunk=250):
        """Minimum over a dense sampling of both segments, as an oracle.

        Slow and obvious on purpose: it shares no code with the vectorized
        implementation, so agreeing with it means something. Sampled in chunks
        along A because the full ``samples x samples x 3`` difference array is
        over a hundred mebibytes, which is a lot to allocate for one assertion.
        """
        params = np.linspace(0.0, 1.0, samples)
        points_a = np.asarray(start_a, dtype=float) + params[:, None] * (
            np.asarray(end_a, dtype=float) - np.asarray(start_a, dtype=float)
        )
        points_b = np.asarray(start_b, dtype=float) + params[:, None] * (
            np.asarray(end_b, dtype=float) - np.asarray(start_b, dtype=float)
        )
        best = np.inf
        for offset in range(0, samples, chunk):
            deltas = points_a[offset : offset + chunk, None, :] - points_b[None, :, :]
            best = min(best, float(np.min(np.einsum("ijk,ijk->ij", deltas, deltas))))
        return best

    def _assert_matches_brute_force(self, start_b, end_b, expected_region):
        start_a, end_a = self.SEGMENT_A

        self.assertEqual(
            self._where_the_minimum_lies(start_a, end_a, start_b, end_b),
            expected_region,
            "this case no longer has the geometry it was chosen for",
        )
        actual = self._distance_squared(start_a, end_a, [start_b], [end_b])[0]

        np.testing.assert_allclose(
            actual, self._brute_force(start_a, end_a, start_b, end_b), rtol=2e-4
        )

    def test_closest_point_beyond_the_far_end_of_the_balloon_sweep(self):
        # Correct answer 0.7299; the near-end formula gives 50.19, a factor of 69.
        self._assert_matches_brute_force(*self.CLAMPED_HIGH, "t_too_high")

    def test_closest_point_before_the_near_end_of_the_balloon_sweep(self):
        # Correct answer 3.4073; the far-end formula gives 80.24, a factor of 24.
        self._assert_matches_brute_force(*self.CLAMPED_LOW, "t_too_low")

    def test_parallel_sweeps_with_this_rocket_segment(self):
        self._assert_matches_brute_force(*self.PARALLEL, "parallel")

    def test_a_balloon_that_did_not_move_over_the_step(self):
        self._assert_matches_brute_force(*self.DEGENERATE_B, "degenerate_b")

    def test_a_balloon_that_barely_moved_but_not_along_the_rocket(self):
        """A near-degenerate sweep where the denominator is *not* near zero.

        The degenerate-B branch is selected on ``e_coeff <= epsilon``, and the
        regular branch is masked to exclude those rows. For a balloon that did not
        move at all the denominator collapses to zero too, so dropping that mask
        changes nothing and the bug hides. Sitting the balloon exactly
        perpendicular to the rocket does not help either: the two formulas are
        algebraically equal when ``direction_a`` and the balloon's step are
        orthogonal, which is how a first attempt at this test reached the branch
        and still passed against the mutation.

        What separates them is a *small, oblique* step. ``e_coeff`` is 1e-12, at
        the threshold, while ``a_coeff * e_coeff - b_coeff**2`` is 1.4e-11, above
        it, so the unmasked regular branch claims the row: it puts the rocket-side
        parameter at 0.0 where the degenerate formula puts it at 0.95, and leaves
        ``t`` at zero because that assignment *is* masked. The reported squared
        distance goes from 21.1 to 76.8.

        Physically this is a balloon hanging almost still over one 0.01 s step,
        which scenario #1 produces in quantity.
        """
        start_a, end_a = self.SEGMENT_A
        start_a, end_a = self.SEGMENT_A
        start_b, end_b = self.NEAR_DEGENERATE_B
        step = np.asarray(end_b) - np.asarray(start_b)
        # Oblique, not orthogonal: the two formulas agree when it is orthogonal.
        self.assertNotEqual(float((end_a - start_a) @ step), 0.0)
        self.assertLess(float(np.linalg.norm(step)), 1e-6)

        self._assert_matches_brute_force(start_b, end_b, "degenerate_b")

    def test_the_rocket_side_parameter_is_clamped_too(self):
        """The other clamp, on s rather than t.

        The two tests above clamp the balloon parameter and recompute the rocket
        one. This is the reverse: the closest point on the rocket sweep lies far
        outside it while the balloon parameter stays inside, so s is clamped and
        the t recomputation never runs. Dropping that clip reports 5.7e8 instead
        of 54.3, and every other case here passes without it because their s
        already lands in range or a t clamp overwrites it.
        """
        self._assert_matches_brute_force(*self.S_CLAMPED, "s_clamped")

    def test_a_mixed_batch_takes_every_branch_at_once(self):
        """All four branches in one call, because they are masks over one array.

        The implementation masks the branches rather than looping, so a
        per-case test cannot see them interact: a mask written with the wrong
        index, or a clamp applied to the whole array instead of its subset,
        needs more than one branch present at once to show up.
        """
        start_a, end_a = self.SEGMENT_A
        cases = [
            ("degenerate_b", *self.DEGENERATE_B),
            ("parallel", *self.PARALLEL),
            ("t_too_high", *self.CLAMPED_HIGH),
            ("t_too_low", *self.CLAMPED_LOW),
            # The near-degenerate row belongs here rather than only in its own
            # test. The regular-branch block is guarded by `if np.any(regular)`,
            # so on its own that row skips the block entirely and a mask written
            # without its `regular &` term never runs. It needs a regular row
            # beside it to fire.
            ("degenerate_b", *self.NEAR_DEGENERATE_B),
            ("s_clamped", *self.S_CLAMPED),
        ]
        # The point of the batch, asserted rather than assumed.
        for expected_region, start_b, end_b in cases:
            self.assertEqual(
                self._where_the_minimum_lies(start_a, end_a, start_b, end_b),
                expected_region,
                f"the batch no longer covers {expected_region} via {start_b}",
            )

        actual = self._distance_squared(
            start_a, end_a, [c[1] for c in cases], [c[2] for c in cases]
        )

        expected = [
            self._brute_force(start_a, end_a, start_b, end_b)
            for _, start_b, end_b in cases
        ]
        np.testing.assert_allclose(actual, expected, rtol=2e-4)

    def test_reordering_the_batch_only_reorders_the_result(self):
        """The masks must follow the rows, not fixed positions.

        Every branch above lands on a different row here, so a clamp that wrote
        to the wrong slice would agree with the batch above and disagree here.
        """
        start_a, end_a = self.SEGMENT_A
        pairs = [
            self.CLAMPED_LOW,
            self.NEAR_DEGENERATE_B,
            self.PARALLEL,
            self.DEGENERATE_B,
            self.CLAMPED_HIGH,
            self.S_CLAMPED,
        ]

        forward = self._distance_squared(
            start_a, end_a, [p[0] for p in pairs], [p[1] for p in pairs]
        )
        reversed_pairs = pairs[::-1]
        backward = self._distance_squared(
            start_a,
            end_a,
            [p[0] for p in reversed_pairs],
            [p[1] for p in reversed_pairs],
        )

        np.testing.assert_allclose(forward, backward[::-1], rtol=0, atol=0)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheSweptPathRule(unittest.TestCase):
    """The scoring rule settled in #63, pinned so a change has to be deliberate.

    The maintainer chose to compare the two swept paths independently rather than
    at matching instants. That is a scoring decision, not an approximation, so a
    later change to same-instant collision would silently move every score. This
    is the case that separates the two.
    """

    def test_paths_crossing_at_different_times_still_pop(self):
        # The rocket reaches x = 9 at 90% of the timestep; the balloon is there at
        # 50%. They never share the point at the same moment.
        rocket_start = np.array([0.0, 0.0, 0.0])
        rocket_end = np.array([10.0, 0.0, 0.0])
        balloon_start = np.array([[9.0, -20.0, 0.0]])
        balloon_end = np.array([[9.0, 20.0, 0.0]])
        radius = 1.5

        # Through _detect_pops, not the geometry helper: the rule is the pop, and
        # the helper is one step of it. Asserting on the distance alone would
        # still pass if the radius comparison or the status write changed.
        env = TestDetectPops._env_with_state(
            balloon_end, [TestDetectPops.RELEASED], rocket_now=rocket_end, radius=radius
        )
        env._detect_pops(
            previous_balloon_positions=balloon_start,
            previous_rocket_position=rocket_start,
        )

        self.assertEqual(
            env._balloon_status[0, 0],
            TestDetectPops.POPPED,
            "the swept rule should register a pop",
        )

        # The case only means something if a same-instant rule would miss it.
        params = np.linspace(0.0, 1.0, 20001)
        rocket = rocket_start + params[:, None] * (rocket_end - rocket_start)
        balloon = balloon_start[0] + params[:, None] * (
            balloon_end[0] - balloon_start[0]
        )
        same_instant = float(np.min(np.sum((rocket - balloon) ** 2, axis=1)))

        self.assertGreater(same_instant, radius**2)

    def test_the_same_geometry_one_timestep_earlier_does_not_pop(self):
        """The rule is not "any two paths that ever cross".

        The pair above pops because both sweeps happen to overlap in space within
        the same step. Shift the balloon's sweep to the step before and the same
        two trajectories no longer produce a pop, which is what makes the rule a
        per-step comparison rather than a whole-flight one.
        """
        rocket_start = np.array([0.0, 0.0, 0.0])
        rocket_end = np.array([10.0, 0.0, 0.0])
        # The balloon crosses x = 9 forty metres away in y during this step.
        balloon_start = np.array([[9.0, -60.0, 0.0]])
        balloon_end = np.array([[9.0, -20.0, 0.0]])

        env = TestDetectPops._env_with_state(
            balloon_end, [TestDetectPops.RELEASED], rocket_now=rocket_end, radius=1.5
        )
        env._detect_pops(
            previous_balloon_positions=balloon_start,
            previous_rocket_position=rocket_start,
        )

        self.assertEqual(env._balloon_status[0, 0], TestDetectPops.RELEASED)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheRewardContract(unittest.TestCase):
    """What ``step`` returns, which the helper tests above never reach.

    The tests above call ``_detect_pops`` directly. They say nothing about
    whether ``step`` wires it up, what it returns, or whether a balloon can be
    scored twice. That last one decides the competition, so it is worth pinning
    rather than assuming.
    """

    @staticmethod
    def _idle(env):
        action = env.action_space.sample()
        action["launch"] = np.array(0, dtype=action["launch"].dtype)
        for key in ("tvc", "throttle", "roll"):
            action[key] = np.zeros_like(action[key])
        return action

    def _launched(self):
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        parameters, _ = load_scenario_parameters(0)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        env.reset(seed=parameters["scenario"]["random_seed"])
        launch = self._idle(env)
        launch["launch"] = np.array(1, dtype=launch["launch"].dtype)
        launch["launch_inclination_heading"] = np.array(
            [90.0, 0.0], dtype=launch["launch_inclination_heading"].dtype
        )
        env.step(launch)
        return env

    @staticmethod
    def _park(env, on_the_rocket):
        """Move ``on_the_rocket`` balloons onto the launch point, the rest far away.

        ``_balloon_states`` is set as well as ``_balloon_flights``. ``step`` copies
        the previous balloon positions out of ``_balloon_states`` *before* refilling
        it from ``_balloon_flights``, so writing only the flights would leave the
        first step sweeping from wherever the balloon really was to the parked
        position. That sweep still crosses the rocket, so the test passed while
        measuring geometry it did not describe.
        """
        here = np.asarray(env.initial_solution[1:4], dtype=float)
        far = here + np.array([0.0, 0.0, 10_000.0])
        env._balloon_flights[:, :3, :] = far[None, :, None]
        env._balloon_flights[:on_the_rocket, :3, :] = here[:, None]
        env._balloon_flights[:, 3:, :] = 0.0
        env._balloon_states = env._balloon_flights[:, :, env.current_step].copy()
        env._balloon_status[:, 0] = 1
        return here

    def test_reward_is_the_step_delta_and_a_pop_never_scores_twice(self):
        env = self._launched()

        # One balloon parked on the launch point, the rest far enough away that
        # only the first can account for any reward.
        self._park(env, on_the_rocket=1)

        _obs, first_reward, _term, _trunc, first_info = env.step(self._idle(env))
        _obs, second_reward, _term, _trunc, second_info = env.step(self._idle(env))

        self.assertEqual(first_reward, 1, "one new pop is a reward of exactly one")
        self.assertEqual(first_info["popped_count"], 1)
        # The balloon is still sitting on the rocket, so the only thing stopping
        # it scoring again is the status check.
        self.assertEqual(second_reward, 0, "an already popped balloon scored again")
        self.assertEqual(second_info["popped_count"], 1, "the count must not drift")

    def test_three_balloons_popped_in_one_step_score_three(self):
        """The reward is the *size* of the delta, not that a delta happened.

        Every other reward test here pops exactly one balloon per step, so
        ``reward = int(new_count > previous_count)`` satisfies all of them while
        capping a multi-pop step at one point. Scenario #1 releases a hundred
        balloons into a ~0.01 s step, so more than one pop per step is reachable
        and the cap would quietly lower a competitor's score.
        """
        env = self._launched()
        parked = 3
        self._park(env, on_the_rocket=parked)

        _obs, reward, _term, _trunc, info = env.step(self._idle(env))

        self.assertEqual(reward, parked, "the reward must count every new pop")
        self.assertEqual(info["popped_count"], parked)
        # Same balloons, still on the rocket: the second step adds nothing.
        _obs, again, _term, _trunc, later = env.step(self._idle(env))
        self.assertEqual(again, 0)
        self.assertEqual(later["popped_count"], parked)

    def test_the_reward_sums_to_the_final_popped_count(self):
        """The cumulative count is the score, and the reward is what an agent sees.

        They are computed differently, so pin that they agree over a whole run
        rather than only at the step boundary each test above happens to check.
        """
        env = self._launched()
        self._park(env, on_the_rocket=2)

        total = 0
        info = {"popped_count": 0}
        for _ in range(5):
            _obs, reward, terminated, truncated, info = env.step(self._idle(env))
            total += reward
            if terminated or truncated:
                break

        self.assertEqual(total, info["popped_count"])
        self.assertEqual(total, 2)

    def test_a_distant_balloon_earns_nothing(self):
        env = self._launched()
        self._park(env, on_the_rocket=0)

        _obs, reward, _term, _trunc, info = env.step(self._idle(env))

        self.assertEqual(reward, 0)
        self.assertEqual(info["popped_count"], 0)


if __name__ == "__main__":
    unittest.main()
