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
        terminated = truncated = False
        steps = 0
        # Before the fix this raised AttributeError on the final step, because
        # the timeout branch post-processed a flight that was never created.
        while not (terminated or truncated):
            _observation, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            self.assertLess(steps, env.num_timesteps + 5, "episode did not end")

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

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            observation, _reward, terminated, truncated, _info = env.step(
                agent.get_action(observation)
            )
            steps += 1
            self.assertLess(steps, env.num_timesteps + 5)

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

        # Without the launch-state fallback the sweep starts from NaN, every
        # comparison is false and nothing pops.
        self.assertGreater(reward, 0, "the first simulated interval registered no pop")
        self.assertGreater(info["popped_count"], 0)


if __name__ == "__main__":
    unittest.main()
