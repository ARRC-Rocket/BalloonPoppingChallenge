"""Episode lifecycle tests for ``BalloonPoppingEnv.step``.

Two paths that a competitor can hit but no test covered:

* an agent is free never to send a launch action, so the episode has to reach
  its timeout and terminate instead of raising;
* the rocket flight is built on the launch action and only produces a state on
  the step after it, so the launch-to-first-sample interval has to take part in
  pop detection rather than being silently skipped.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import unittest

from tests.bounded_episode import run_episode
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0


def _idle_action(env):
    """A well-formed action that never launches and commands nothing."""
    action = env.action_space.sample()
    action["launch"] = np.array(0, dtype=action["launch"].dtype)
    action["tvc"] = np.zeros_like(action["tvc"])
    action["throttle"] = np.zeros_like(action["throttle"])
    action["roll"] = np.zeros_like(action["roll"])
    return action


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestNeverLaunchingAgent(unittest.TestCase):
    """An agent that never launches must time out, not raise."""

    def test_episode_times_out_without_a_launch(self):
        scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        env.reset(seed=scenario_params["scenario"]["random_seed"])

        action = _idle_action(env)
        # Before the fix this raised AttributeError on the final step, because
        # the timeout branch post-processed a flight that was never created.
        steps, terminated, truncated, info = run_episode(env, lambda _o: action)

        # Running out of horizon is truncation, not termination: nothing about
        # the rocket ended the episode, the precomputed clock did. Reporting it
        # as terminated tells an algorithm not to bootstrap the final value,
        # which for an agent that simply never launched is the wrong lesson.
        self.assertFalse(terminated, "the flight did not end; the clock did")
        self.assertTrue(truncated)
        self.assertEqual(steps, env.num_timesteps - 1)

        self.assertIsNone(env._rocket_flight, "no flight should exist without a launch")
        self.assertEqual(info["popped_count"], 0)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestWhatATruncatedEpisodeReports(unittest.TestCase):
    """Two things the split between the two causes left behind.

    Splitting ``terminated`` from ``truncated`` was correct, and both of these
    read ``terminated`` back when that one flag also covered the clock. Neither
    moved with it.
    """

    @classmethod
    def setUpClass(cls):
        cls.scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)

    def _run_to_the_horizon(self, env):
        action = _idle_action(env)
        _steps, terminated, truncated, _info = run_episode(env, lambda _o: action)
        return terminated, truncated

    def test_running_out_of_time_is_logged_as_truncation(self):
        """It still said "Terminated: Reached max time".

        Which is the one line an operator reads to find out why a run stopped,
        and it named the case the flags now deliberately do not.
        """
        env = BalloonPoppingEnv(render_mode=None, parameters=self.scenario_params)
        env.reset(seed=self.scenario_params["scenario"]["random_seed"])

        with self.assertLogs("BalloonPoppingGymEnv", level="INFO") as captured:
            _terminated, truncated = self._run_to_the_horizon(env)

        self.assertTrue(truncated)
        messages = [record.getMessage() for record in captured.records]
        self.assertIn("Truncated: Reached max simulation time", messages)
        self.assertNotIn("Terminated: Reached max time", messages)

    def test_the_last_frame_of_a_truncated_episode_is_drawn(self):
        """Frames are drawn on a 0.1 s cadence, or when the episode ends.

        While the clock counted as termination, an episode that ran out of
        horizon always drew its final frame. Afterwards the gate only fired on
        ``terminated``, so a truncated episode whose last step misses the
        cadence stopped drawing one. Scenario 1 ends this way normally.
        """
        env = BalloonPoppingEnv(render_mode=None, parameters=self.scenario_params)
        env.reset(seed=self.scenario_params["scenario"]["random_seed"])

        drawn_at = []
        original = env._render_frame
        env._render_frame = lambda *args, **kwargs: (
            drawn_at.append(env.current_step),
            original(*args, **kwargs),
        )[1]

        _terminated, truncated = self._run_to_the_horizon(env)

        self.assertTrue(truncated)
        cadence = int(0.1 / self.scenario_params["simulation"]["time_step"])
        # Or the old gate would have drawn this frame anyway and the assertion
        # below would hold whichever version is in place.
        self.assertNotEqual(
            env.current_step % cadence,
            0,
            "the final step lands on the render cadence, so this cannot tell "
            "the two versions apart",
        )
        self.assertEqual(drawn_at[-1], env.current_step)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestAFinishedFlightTerminates(unittest.TestCase):
    """The other end of the pair, so neither flag is simply hard-coded.

    Without this, setting terminated to False unconditionally would satisfy the
    timeout case above and nothing else would notice.
    """

    def test_a_flight_that_lands_terminates_rather_than_truncating(self):
        scenario_params, given = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        agent = AttitudeRateControlAgent(
            given, rate_targets=[0.0, 0.0, 0.0], launch_time=1
        )
        observation, _ = env.reset(seed=scenario_params["scenario"]["random_seed"])

        steps, terminated, truncated, _info = run_episode(
            env, agent.get_action, observation
        )

        self.assertTrue(terminated, "the flight finished, so the episode ended")
        self.assertFalse(truncated)
        # And it really did stop early rather than reaching the horizon.
        self.assertLess(steps, env.num_timesteps - 1)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestFirstPostLaunchInterval(unittest.TestCase):
    """The launch-to-first-sample interval takes part in pop detection."""

    def test_balloon_on_the_first_interval_is_popped(self):
        scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        env.reset(seed=scenario_params["scenario"]["random_seed"])

        launch = _idle_action(env)
        launch["launch"] = np.array(1, dtype=launch["launch"].dtype)
        launch["launch_inclination_heading"] = np.array(
            [90.0, 0.0], dtype=launch["launch_inclination_heading"].dtype
        )
        env.step(launch)

        # The flight exists but has not been stepped, so the rocket is still at
        # its launch state and the recorded state array is still all-NaN.
        self.assertIsNotNone(env._rocket_flight)
        self.assertTrue(np.isnan(env._rocket_states[0]))

        # Park a released balloon on the launch position for the whole horizon,
        # so only the first simulated interval can account for a pop.
        launch_position = np.asarray(env.initial_solution[1:4], dtype=float)
        env._balloon_flights[:, :3, :] = launch_position[None, :, None]
        env._balloon_flights[:, 3:, :] = 0.0
        env._balloon_status[:, 0] = 1

        _observation, reward, _terminated, _truncated, info = env.step(
            _idle_action(env)
        )

        # Without seeding the sweep origin from the launch state there is no
        # valid start for this interval and nothing pops.
        self.assertGreater(reward, 0, "the first simulated interval registered no pop")
        self.assertGreater(info["popped_count"], 0)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestSweepOriginAdvances(unittest.TestCase):
    """Each sweep starts where the previous one ended, not at the pad.

    The first-interval test above only pins the launch-time initialization: the
    balloons it parks never move, so it still passes if the origin is left
    stuck at the launch point. The golden masters do not catch that either,
    since scenario 0 pops all ten balloons regardless and scenario 1 pops none.
    """

    def _launched_env(self):
        scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        env.reset(seed=scenario_params["scenario"]["random_seed"])
        launch = _idle_action(env)
        launch["launch"] = np.array(1, dtype=launch["launch"].dtype)
        launch["launch_inclination_heading"] = np.array(
            [90.0, 0.0], dtype=launch["launch_inclination_heading"].dtype
        )
        env.step(launch)
        return env

    def test_origin_starts_at_the_launch_state(self):
        env = self._launched_env()
        np.testing.assert_allclose(
            env._sweep_origin, np.asarray(env.initial_solution[1:4], dtype=float)
        )

    def test_origin_is_cleared_by_reset(self):
        env = self._launched_env()
        self.assertIsNotNone(env._sweep_origin)
        scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env.reset(seed=scenario_params["scenario"]["random_seed"])
        self.assertIsNone(env._sweep_origin)

    def test_second_interval_sweeps_from_the_first_endpoint(self):
        env = self._launched_env()
        launch_position = np.asarray(env.initial_solution[1:4], dtype=float)

        env.step(_idle_action(env))
        first_endpoint = env._rocket_states[:3].copy()
        # The rocket has to have moved, or the two candidates are the same point
        # and the assertion below would hold either way.
        self.assertGreater(
            float(np.linalg.norm(first_endpoint - launch_position)),
            1e-9,
            "the rocket did not move, so this test cannot discriminate",
        )

        swept_from = []
        original = env._detect_pops

        def recording_detect_pops(previous_balloon_positions, previous_rocket_position):
            swept_from.append(np.asarray(previous_rocket_position, dtype=float).copy())
            return original(previous_balloon_positions, previous_rocket_position)

        env._detect_pops = recording_detect_pops
        env.step(_idle_action(env))

        self.assertEqual(len(swept_from), 1, "one sweep per flown step")
        # Deleting the advancement leaves the origin at the pad, which this
        # catches; the first-interval test above does not.
        np.testing.assert_allclose(swept_from[0], first_endpoint)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheWallClockLimit(unittest.TestCase):
    """The limit is on how long an episode takes to run, not to simulate.

    It rules out algorithms that cannot keep up in real time, so it ends the
    episode the way the horizon does rather than the way a finished flight does.
    """

    def _parameters(self, wall_limit):
        parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        if wall_limit is None:
            parameters["simulation"].pop("max_wall_time", None)
        else:
            parameters["simulation"]["max_wall_time"] = wall_limit
        return parameters

    def _idle_episode(self, parameters):
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        env.reset(seed=parameters["scenario"]["random_seed"])
        action = _idle_action(env)
        return env, lambda: run_episode(env, lambda _o: action)

    def test_an_episode_over_the_limit_is_truncated(self):
        """A limit far shorter than the run has to end it, and did not.

        ``reset`` started the clock on ``time.time`` while ``step`` read
        ``time.monotonic``, so the difference was about -1.8e9 on every step and
        no episode could reach any limit.
        """
        env, run = self._idle_episode(self._parameters(0.05))

        steps, terminated, truncated, _info = run()

        self.assertTrue(truncated)
        self.assertFalse(terminated, "the flight did not end; the clock did")
        self.assertLess(steps, env.num_timesteps - 1, "it ran to the horizon instead")

    def test_the_truncation_says_which_clock_ended_it(self):
        """One line separates the two truncations for whoever reads the log."""
        _env, run = self._idle_episode(self._parameters(0.05))

        with self.assertLogs("BalloonPoppingGymEnv", level="INFO") as captured:
            run()

        messages = [record.getMessage() for record in captured.records]
        self.assertTrue(
            any(m.startswith("Truncated: Reached max wall time") for m in messages),
            messages,
        )
        self.assertNotIn("Truncated: Reached max simulation time", messages)

    def test_a_scenario_without_the_key_keeps_running(self):
        """The control. A scenario file written before this is not limited."""
        env, run = self._idle_episode(self._parameters(None))

        steps, _terminated, truncated, _info = run()

        self.assertTrue(truncated)
        self.assertEqual(steps, env.num_timesteps - 1)


if __name__ == "__main__":
    unittest.main()
