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

    def test_degenerate_segment_b_is_a_point(self):
        # B collapses to (0, 4, 0); nearest point on A is the origin -> 16.
        result = self._distance_squared([-1, 0, 0], [1, 0, 0], [[0, 4, 0]], [[0, 4, 0]])
        np.testing.assert_allclose(result, [16.0], atol=1e-9, rtol=0)

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


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheClampingBranches(unittest.TestCase):
    """The endpoint clamping the fixed cases above do not reach.

    When the closest point on a balloon sweep falls beyond its far endpoint, the
    parameter is clamped and the rocket-side parameter is recomputed against that
    endpoint. Swapping that recomputation for the near-endpoint one leaves every
    other test in this file green, so it needs its own case (see #63 discussion).
    """

    def _distance_squared(self, start_a, end_a, starts_b, ends_b):
        return BalloonPoppingEnv._segment_distance_squared_batch(
            start_a, end_a, starts_b, ends_b
        )

    @staticmethod
    def _brute_force(start_a, end_a, start_b, end_b, samples=2001):
        """Minimum over a dense sampling of both segments, as an oracle.

        Slow and obvious on purpose: it shares no code with the vectorized
        implementation, so agreeing with it means something.
        """
        params = np.linspace(0.0, 1.0, samples)
        points_a = np.asarray(start_a) + params[:, None] * (
            np.asarray(end_a) - np.asarray(start_a)
        )
        points_b = np.asarray(start_b) + params[:, None] * (
            np.asarray(end_b) - np.asarray(start_b)
        )
        deltas = points_a[:, None, :] - points_b[None, :, :]
        return float(np.min(np.einsum("ijk,ijk->ij", deltas, deltas)))

    def test_closest_point_beyond_the_far_end_of_the_balloon_sweep(self):
        # Chosen by searching for a pair where the far-end and near-end
        # recomputations disagree by a wide margin, rather than by picking
        # something that merely reaches the branch: my first attempt reached it
        # and still clamped both to the same value, so the mutation survived.
        # Here the correct answer is 1.958 and the near-end formula gives 75.53.
        start_a, end_a = [0.0, 0.0, 0.0], [-4.143, -4.990, -4.617]
        start_b, end_b = [-7.559, 7.589, 6.488], [-5.483, -5.211, -4.280]

        actual = self._distance_squared(start_a, end_a, [start_b], [end_b])[0]

        np.testing.assert_allclose(
            actual, self._brute_force(start_a, end_a, start_b, end_b), rtol=2e-4
        )

    def test_closest_point_before_the_near_end_of_the_balloon_sweep(self):
        # Same treatment for the other clamp: correct is 4.854, the far-end
        # formula gives 86.73.
        start_a, end_a = [0.0, 0.0, 0.0], [-4.911, -4.398, -4.254]
        start_b, end_b = [-6.734, -5.388, -3.514], [-6.209, 0.970, 7.798]

        actual = self._distance_squared(start_a, end_a, [start_b], [end_b])[0]

        np.testing.assert_allclose(
            actual, self._brute_force(start_a, end_a, start_b, end_b), rtol=2e-4
        )

    def test_a_mixed_batch_takes_every_branch_at_once(self):
        # Degenerate, parallel, skew clamped high and skew clamped low, in one
        # call, because the implementation masks the branches rather than
        # looping and a per-case test cannot see them interact.
        start_a, end_a = [0.0, 0.0, 0.0], [-4.5, -4.7, -4.4]
        starts_b = [
            [2.0, 3.0, 0.0],
            [0.0, 5.0, 0.0],
            [-7.559, 7.589, 6.488],
            [-6.734, -5.388, -3.514],
        ]
        ends_b = [
            [2.0, 3.0, 0.0],
            [4.0, 5.0, 0.0],
            [-5.483, -5.211, -4.280],
            [-6.209, 0.970, 7.798],
        ]

        actual = self._distance_squared(start_a, end_a, starts_b, ends_b)

        expected = [
            self._brute_force(start_a, end_a, s, e)
            for s, e in zip(starts_b, ends_b, strict=True)
        ]
        np.testing.assert_allclose(actual, expected, rtol=2e-4)


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
        rocket_start, rocket_end = np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])
        balloon_start = np.array([[9.0, -20.0, 0.0]])
        balloon_end = np.array([[9.0, 20.0, 0.0]])

        swept = BalloonPoppingEnv._segment_distance_squared_batch(
            rocket_start, rocket_end, balloon_start, balloon_end
        )[0]

        # Same-instant distance, which is what a synchronized rule would use.
        params = np.linspace(0.0, 1.0, 20001)
        rocket = rocket_start + params[:, None] * (rocket_end - rocket_start)
        balloon = balloon_start[0] + params[:, None] * (
            balloon_end[0] - balloon_start[0]
        )
        same_instant = float(np.min(np.sum((rocket - balloon) ** 2, axis=1)))

        radius = 1.5
        self.assertLessEqual(swept, radius**2, "the swept rule should register a pop")
        self.assertGreater(
            same_instant,
            radius**2,
            "the case is only meaningful if a same-instant rule would miss it",
        )


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

    def test_reward_is_the_step_delta_and_a_pop_never_scores_twice(self):
        env = self._launched()

        # One balloon parked on the launch point, the rest far enough away that
        # only the first can account for any reward.
        here = np.asarray(env.initial_solution[1:4], dtype=float)
        env._balloon_flights[:, :3, :] = (here + np.array([0.0, 0.0, 10_000.0]))[
            None, :, None
        ]
        env._balloon_flights[0, :3, :] = here[:, None]
        env._balloon_flights[:, 3:, :] = 0.0
        env._balloon_status[:, 0] = 1

        _obs, first_reward, _term, _trunc, first_info = env.step(self._idle(env))
        _obs, second_reward, _term, _trunc, second_info = env.step(self._idle(env))

        self.assertEqual(first_reward, 1, "one new pop is a reward of exactly one")
        self.assertEqual(first_info["popped_count"], 1)
        # The balloon is still sitting on the rocket, so the only thing stopping
        # it scoring again is the status check.
        self.assertEqual(second_reward, 0, "an already popped balloon scored again")
        self.assertEqual(second_info["popped_count"], 1, "the count must not drift")

    def test_a_distant_balloon_earns_nothing(self):
        env = self._launched()
        here = np.asarray(env.initial_solution[1:4], dtype=float)
        env._balloon_flights[:, :3, :] = (here + np.array([0.0, 0.0, 10_000.0]))[
            None, :, None
        ]
        env._balloon_flights[:, 3:, :] = 0.0
        env._balloon_status[:, 0] = 1

        _obs, reward, _term, _trunc, info = env.step(self._idle(env))

        self.assertEqual(reward, 0)
        self.assertEqual(info["popped_count"], 0)
