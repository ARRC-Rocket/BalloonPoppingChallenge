"""The launched flight has to stay JSON serializable for the submission packer.

``pack_for_submission`` runs the flight through ``RocketPyEncoder``. The flight
carries the rocket, the rocket carries its sensors, and a sensor hands back
whatever seed it was built with. A seed that has no JSON form therefore breaks
submission packing for every competitor, which is exactly what a ``SeedSequence``
seed did. Nothing covered that path before, on either side of the submodule.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import copy
import json
import os
import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from rocketpy._encoders import RocketPyEncoder

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0

# Running an episode to termination costs about half a minute, so the tests
# that need a finished flight sit behind the same gate as the golden masters.
_RUN_SLOW = os.environ.get("BPC_RUN_SLOW_TESTS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _example_agent(given_parameters):
    """The shipped example agent, so the flight is a realistic one."""
    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent

    return AttitudeRateControlAgent(given_parameters)


def _launched_env(seed=None, np_random=None):
    """Build scenario 0 and send the launch action so the sensors exist."""
    parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    if np_random is not None:
        env.np_random = np_random
        env.reset()
    else:
        if seed is None:
            seed = parameters["scenario"]["random_seed"]
        env.reset(seed=seed)

    action = env.action_space.sample()
    action["launch"] = np.array(1, dtype=action["launch"].dtype)
    action["launch_inclination_heading"] = np.array(
        [90.0, 0.0], dtype=action["launch_inclination_heading"].dtype
    )
    for key in ("tvc", "throttle", "roll"):
        action[key] = np.zeros_like(action[key])
    env.step(action)
    return env


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestSensorSeedsAreJsonSafe(unittest.TestCase):
    def test_sensor_seeds_are_plain_integers(self):
        env = _launched_env()
        sensors = env._rocket_flight.rocket.sensors
        self.assertTrue(sensors, "the launched rocket should carry sensors")
        for component in sensors:
            sensor = component[0] if isinstance(component, (tuple, list)) else component
            self.assertIsInstance(
                sensor._seed, int, "a sensor seed has to survive JSON encoding"
            )

    def test_sensor_dicts_survive_the_submission_encoder(self):
        """The check that fails on a SeedSequence seed, without a full flight."""
        env = _launched_env()
        for component in env._rocket_flight.rocket.sensors:
            sensor = component[0] if isinstance(component, (tuple, list)) else component
            json.dumps(sensor.to_dict(), cls=RocketPyEncoder, allow_pickle=False)

    def test_seeds_are_reproducible_and_distinct(self):
        first = _launched_env(seed=7)
        second = _launched_env(seed=7)
        other = _launched_env(seed=8)

        def seeds(env):
            return [
                (c[0] if isinstance(c, (tuple, list)) else c)._seed
                for c in env._rocket_flight.rocket.sensors
            ]

        self.assertEqual(seeds(first), seeds(second), "same seed must replay")
        self.assertNotEqual(seeds(first), seeds(other), "a new seed must differ")
        self.assertEqual(
            len(set(seeds(first))), len(seeds(first)), "sensors must not share a seed"
        )


@unittest.skipUnless(_RUN_SLOW, "slow: runs a scenario 0 episode to termination")
class TestTheProductionBoundary(unittest.TestCase):
    """What ``pack_for_submission`` actually encodes, not a piece of it.

    The tests above stop at ``sensor.to_dict()``. The packer encodes the whole
    flight, so the failure it hit reached the sensor through Flight, Rocket and
    the encoder's circular-reference handling. Testing a leaf leaves every layer
    above it uncovered, and the next value with no JSON form will sit somewhere
    else in that graph.

    The episode runs to termination rather than stopping after the launch step,
    because that is the state the packer is called in. An unfinished flight has
    no ``t_final`` and its ``to_dict()`` raises AttributeError, which says
    nothing about whether a submission can be packed.
    """

    @classmethod
    def setUpClass(cls):
        parameters, given = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        agent = _example_agent(given)
        observation, _ = env.reset(seed=parameters["scenario"]["random_seed"])
        terminated = False
        while not terminated:
            observation, _reward, terminated, _truncated, _info = env.step(
                agent.get_action(observation)
            )
        cls.env = env

    def test_the_whole_flight_survives_the_encoder(self):
        # This exact call is what raised TypeError on a SeedSequence seed.
        json.dumps(self.env._rocket_flight, cls=RocketPyEncoder)

    def test_the_real_packer_writes_a_readable_submission(self):
        """Drives ``pack_for_submission`` itself on a finished flight.

        Nothing else covers the packer against a real one: the other tests that
        call it hand it ``_rocket_flight=None``.
        """
        import glob
        import pickle
        from unittest import mock

        from BalloonPoppingGymEnv.evaluation.results import utils

        results_dir = os.path.dirname(os.path.abspath(utils.__file__))
        agent_path = os.path.join(results_dir, "_serialization_test_agent.py")
        with open(agent_path, "w", encoding="utf-8") as handle:
            handle.write("# test agent source\n")

        eval_cfg = {
            "team_name": "unittest_team",
            "team_secret": "secret",
            "agent_name": "agent",
            "scenario_number": SCENARIO_NUMBER,
            "agent_module_path": agent_path,
        }
        pattern = os.path.join(results_dir, "*_submission.*")
        before = set(glob.glob(pattern))
        try:
            with mock.patch.object(
                utils.urllib.request, "urlopen", side_effect=OSError("offline")
            ):
                utils.pack_for_submission(eval_cfg, self.env, {"scenario": {}})
            written = sorted(set(glob.glob(pattern)) - before)
            self.assertEqual(len(written), 1, "packing should produce one file")
            with open(written[0], "rb") as handle:
                payload = pickle.load(handle)
        finally:
            os.remove(agent_path)
            for path in set(glob.glob(pattern)) - before:
                os.remove(path)

        self.assertEqual(payload["team"]["name"], "unittest_team")
        self.assertIn("rocket_flight", payload["balloon_world_data"])
        self.assertGreater(payload["leaderboard_info"]["final_reward"], 0)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheSeedDerivationIsPinned(unittest.TestCase):
    """The seeds themselves, not just that they are reproducible.

    Converting a spawned child to an int does not preserve the stream the child
    itself would have produced: ``default_rng(child)`` mixes in the child's
    ``spawn_key``, while ``default_rng(int)`` builds a fresh root sequence. That
    was a deliberate change, made so the seed has a JSON form, and both shipped
    scenarios run at ``noise_density: 0.0`` so nothing draws from these
    generators yet.

    Which means nothing would notice the derivation moving again. These values
    are the contract until a noisy scenario ships, and a change to them should be
    a decision rather than a side effect.
    """

    EXPECTED = {
        "gyro": 75104592069875414758853279389992779332,
        "accelerometer": 197454062571126576167403297853671933012,
        "gnss": 121534159824502556988407154930880891588,
    }

    def test_scenario_zero_seeds_have_not_moved(self):
        env = _launched_env(seed=0)
        seeds = [
            (c[0] if isinstance(c, (tuple, list)) else c)._seed
            for c in env._rocket_flight.rocket.sensors
        ]

        self.assertEqual(seeds, list(self.EXPECTED.values()))

    def test_each_seed_still_carries_its_full_width(self):
        # A derivation that quietly dropped to 32 bits would still be
        # reproducible and still be distinct, so width needs its own assertion.
        for name, value in self.EXPECTED.items():
            with self.subTest(sensor=name):
                self.assertGreater(value.bit_length(), 96)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestUnknownSeed(unittest.TestCase):
    """Gymnasium reports -1 when np_random was supplied rather than seeded."""

    def test_launch_works_with_a_directly_assigned_generator(self):
        env = _launched_env(np_random=np.random.default_rng(123))
        self.assertEqual(env.np_random_seed, -1, "Gymnasium reports an unknown seed")
        # Before the fix this raised ValueError from SeedSequence([-1, ...]).
        for component in env._rocket_flight.rocket.sensors:
            sensor = component[0] if isinstance(component, (tuple, list)) else component
            self.assertIsInstance(sensor._seed, int)

    def test_any_negative_seed_is_handled_not_only_minus_one(self):
        """``SeedSequence`` rejects every negative value, not just -1.

        Narrowing the branch to ``== -1`` would turn an unexpected negative into
        a ValueError at launch rather than a usable seed.
        """
        parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        env.reset(seed=0)
        env._np_random_seed = -5

        action = env.action_space.sample()
        action["launch"] = np.array(1, dtype=action["launch"].dtype)
        action["launch_inclination_heading"] = np.array(
            [90.0, 0.0], dtype=action["launch_inclination_heading"].dtype
        )
        for key in ("tvc", "throttle", "roll"):
            action[key] = np.zeros_like(action[key])
        env.step(action)

        for component in env._rocket_flight.rocket.sensors:
            sensor = component[0] if isinstance(component, (tuple, list)) else component
            self.assertIsInstance(sensor._seed, int)
            self.assertGreater(sensor._seed, 0)

    def _launch(self, env):
        action = env.action_space.sample()
        action["launch"] = np.array(1, dtype=action["launch"].dtype)
        action["launch_inclination_heading"] = np.array(
            [90.0, 0.0], dtype=action["launch_inclination_heading"].dtype
        )
        for key in ("tvc", "throttle", "roll"):
            action[key] = np.zeros_like(action[key])
        env.step(action)

    def _next_draw(self, generator):
        return int(generator.integers(0, 2**32, dtype=np.uint32))

    def test_a_known_seed_leaves_the_environment_rng_alone(self):
        """The ordinary path derives from the seed and draws nothing.

        Worth pinning on its own: if seeding the sensors ever started consuming
        the environment generator here, every later draw in the episode would
        shift, and only a golden master would notice.
        """
        parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        env.reset(seed=0)
        untouched = copy.deepcopy(env.np_random)

        self._launch(env)

        self.assertEqual(self._next_draw(env.np_random), self._next_draw(untouched))

    def test_an_unknown_seed_draws_exactly_four_words(self):
        """The fallback path has to take its entropy from somewhere.

        It comes out of the environment generator, which is a real side effect
        rather than a free one. Nothing after launch draws from it today, so it
        costs nothing now; this is what would say so if that changed.
        """
        parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        env.np_random = np.random.default_rng(123)
        env.reset()
        before_launch = copy.deepcopy(env.np_random)

        self._launch(env)

        before_launch.integers(0, 2**32, size=4, dtype=np.uint32)
        self.assertEqual(
            self._next_draw(env.np_random),
            self._next_draw(before_launch),
            "launch should consume exactly four uint32 on the unknown-seed path",
        )


if __name__ == "__main__":
    unittest.main()
