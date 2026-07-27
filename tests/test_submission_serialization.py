"""The launched flight has to stay JSON serializable for the submission packer.

``pack_for_submission`` runs the flight through ``RocketPyEncoder``. The flight
carries the rocket, the rocket carries its sensors, and a sensor hands back
whatever seed it was built with. A seed that has no JSON form therefore breaks
submission packing for every competitor, which is exactly what a ``SeedSequence``
seed did. Nothing covered that path before, on either side of the submodule.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import json
import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from rocketpy._encoders import RocketPyEncoder

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0


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


if __name__ == "__main__":
    unittest.main()
