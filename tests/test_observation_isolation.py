"""What the agent gets back must not be the environment's own state.

``evaluate_scenario`` hands the observation straight to ``agent.get_action()``,
and the agent is a competitor's code. ``step()`` then computes the reward from
``_balloon_status`` after that call returns, and ``pack_for_submission`` records
the resulting count as the score. So an array handed out by reference is not an
API tidiness question, it is the scoring path.

Measured before the fix, on scenario 0: an agent that never sent a launch action
and only wrote 2 into the status array it was given scored 10 out of 10.
"""

import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_0_PARAMS = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_0_parameters.yaml"
)

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import yaml

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheAgentCannotReachTheEnvironment(unittest.TestCase):
    def setUp(self):
        self.parameters = yaml.safe_load(
            SCENARIO_0_PARAMS.read_text(encoding="utf-8-sig")
        )
        self.env = BalloonPoppingEnv(render_mode=None, parameters=self.parameters)
        self.observation, self.info = self.env.reset(
            seed=self.parameters["scenario"]["random_seed"]
        )

    def _idle_action(self):
        action = self.env.action_space.sample()
        action["launch"] = np.array(0, dtype=action["launch"].dtype)
        for key in ("tvc", "roll"):
            action[key] = np.zeros_like(action[key])
        # Held at its initial output rather than zeroed, so this does not also
        # drive the throttle rate limiter.
        action["throttle"] = np.ones_like(action["throttle"])
        return action

    def test_nothing_handed_out_shares_memory_with_the_environment(self):
        """The property, rather than one consequence of losing it.

        Checked on both what reset returns and what step returns, since the two
        go through the same accessors but a future change could make only one of
        them safe.
        """
        pairs = [
            (
                "reset balloon_status",
                self.observation["balloon_status"],
                self.env._balloon_status,
            ),
            (
                "reset balloon_states",
                self.observation["balloon_states"],
                self.env._balloon_states,
            ),
            (
                "reset rocket_sensors",
                self.observation["rocket_sensors"],
                self.env._rocket_sensors,
            ),
            (
                "reset rocket_states",
                self.info["rocket_states"],
                self.env._rocket_states,
            ),
        ]
        observation, _reward, _terminated, _truncated, info = self.env.step(
            self._idle_action()
        )
        pairs += [
            (
                "step balloon_status",
                observation["balloon_status"],
                self.env._balloon_status,
            ),
            (
                "step balloon_states",
                observation["balloon_states"],
                self.env._balloon_states,
            ),
            (
                "step rocket_sensors",
                observation["rocket_sensors"],
                self.env._rocket_sensors,
            ),
            ("step rocket_states", info["rocket_states"], self.env._rocket_states),
        ]

        for label, handed_out, internal in pairs:
            with self.subTest(field=label):
                self.assertFalse(
                    np.shares_memory(handed_out, internal),
                    f"{label} is a writable alias of the environment's state",
                )

    def test_writing_the_status_back_does_not_score(self):
        """The exploit, driven the way an agent would.

        Three lines, no launch, and before the fix it returned the whole
        scenario. The rocket never leaves the pad, so nothing about this run
        could legitimately pop anything.
        """
        self.observation["balloon_status"][:] = 2

        observation, reward, _terminated, _truncated, info = self.env.step(
            self._idle_action()
        )

        self.assertFalse(self.env.rocket_launched, "this run should never launch")
        self.assertEqual(reward, 0)
        self.assertEqual(info["popped_count"], 0)
        self.assertTrue(
            (np.asarray(observation["balloon_status"]) != 2).all(),
            "a balloon is recorded as popped",
        )

    def test_writing_the_balloon_positions_back_does_not_move_them(self):
        """``_balloon_states`` is a slice of the precomputed flight array.

        Handed out by reference, writing to it rewrote the trajectory the pop
        sweep compares the rocket against, which is the same exploit by another
        route: put the balloons on the rocket rather than the score on the
        balloons.
        """
        before = self.env._balloon_flights.copy()

        self.observation["balloon_states"][:] = 12345.0

        np.testing.assert_array_equal(self.env._balloon_flights, before)
        self.assertTrue(
            (np.asarray(self.env._balloon_states) != 12345.0).all(),
            "the environment's balloon states were rewritten",
        )

    def test_writing_the_sensors_and_rocket_state_back_changes_nothing(self):
        observation, _reward, _terminated, _truncated, info = self.env.step(
            self._idle_action()
        )
        sensors_before = self.env._rocket_sensors.copy()
        states_before = self.env._rocket_states.copy()

        observation["rocket_sensors"][:] = 999.0
        info["rocket_states"][:] = 999.0

        np.testing.assert_array_equal(self.env._rocket_sensors, sensors_before)
        np.testing.assert_array_equal(self.env._rocket_states, states_before)

    def test_a_hostile_agent_scores_nothing_over_a_whole_episode(self):
        """End to end, because a single step is not where a score is decided.

        An agent that rewrites everything it is handed on every call, and never
        launches. It should finish with the same score as an agent that does
        nothing at all, which is none.
        """
        action = self._idle_action()
        observation = self.observation
        steps = 0
        while True:
            observation["balloon_status"][:] = 2
            observation["balloon_states"][:] = 0.0
            observation["rocket_sensors"][:] = 0.0
            observation, _reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            if terminated or truncated:
                break
            self.assertLess(steps, self.env.num_timesteps + 5, "episode did not end")

        self.assertEqual(info["popped_count"], 0)
        self.assertFalse(self.env.rocket_launched)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheGivenParametersAreAlsoDetached(unittest.TestCase):
    """The other thing the agent is handed, and it reaches further.

    The observation is what the agent sees every step; ``given_parameters`` is
    what its constructor gets. The extraction builds new dicts, so replacing a
    value there never reached the environment, but any list value was the same
    object the environment reads.

    Measured: four were shared, and two of them are read when the flight is
    built on the launch action, which is after the agent has already run. So an
    agent could widen its own throttle range or change the rocket's moment of
    inertia from its ``__init__``.
    """

    def setUp(self):
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        self.scenario, self.given = load_scenario_parameters(0)

    def test_nothing_in_them_is_the_same_object(self):
        """The property, checked over the whole tree rather than the two known
        offenders, so a new whitelisted key cannot reintroduce this quietly."""
        shared = []

        def walk(given, scenario, path=""):
            if isinstance(given, dict) and isinstance(scenario, dict):
                for key in given:
                    if key in scenario:
                        walk(given[key], scenario[key], f"{path}.{key}")
            elif given is scenario and isinstance(given, (list, dict, set)):
                shared.append(path)

        walk(self.given, self.scenario)

        self.assertEqual(
            shared, [], "given_parameters shares objects with the scenario"
        )

    def test_writing_to_them_does_not_reach_the_environment(self):
        """Driven the way an agent would, on the two that reached the physics."""
        env = BalloonPoppingEnv(render_mode=None, parameters=self.scenario)
        throttle_before = list(env.rocket_parameters["control"]["throttle_range"])
        inertia_before = list(env.rocket_parameters["rocket_body"]["inertia"])

        self.given["rocket"]["control"]["throttle_range"][1] = 99.0
        self.given["rocket"]["rocket_body"]["inertia"][0] = 1e-6

        self.assertEqual(
            list(env.rocket_parameters["control"]["throttle_range"]), throttle_before
        )
        self.assertEqual(
            list(env.rocket_parameters["rocket_body"]["inertia"]), inertia_before
        )

    def test_the_values_are_still_the_ones_the_scenario_holds(self):
        """Or the fix could be "hand the agent something else entirely"."""
        self.assertEqual(
            self.given["rocket"]["control"]["throttle_range"],
            self.scenario["rocket"]["control"]["throttle_range"],
        )
        self.assertEqual(
            self.given["rocket"]["rocket_body"]["inertia"],
            self.scenario["rocket"]["rocket_body"]["inertia"],
        )


if __name__ == "__main__":
    unittest.main()
