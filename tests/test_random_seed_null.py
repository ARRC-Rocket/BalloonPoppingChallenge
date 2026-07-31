"""``random_seed: null`` has to run, because the scenario file offers it.

``scenario_<n>_parameters.yaml`` carries ``random_seed: 0 # Use null to enable
random seeding``, and ``evaluate.py`` passes that value straight to
``env.reset(seed=...)``. ``Env.reset(seed=None)`` leaves ``_np_random`` unset and
only the ``np_random`` property draws a generator, so reading the attribute was
``AttributeError: 'NoneType' object has no attribute 'shuffle'`` on the very
option the comment points at.

Scenario 0 because it skips the Monte Carlo, so this stays a fast test.

Only ``rocketpy`` is guarded: a missing simulation stack is a legitimate skip,
but a broken import inside this package is a failure and must stay loud.
"""

import copy
import unittest
from importlib.util import find_spec

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0


def _parameters(random_seed):
    loaded = load_scenario_parameters(SCENARIO_NUMBER)
    parameters = copy.deepcopy(loaded[0] if isinstance(loaded, tuple) else loaded)
    parameters["scenario"]["random_seed"] = random_seed
    return parameters


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestAnUnseededResetRuns(unittest.TestCase):
    def test_reset_without_a_seed_does_not_raise(self):
        parameters = _parameters(None)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        self.addCleanup(env.close)

        env.reset(seed=parameters["scenario"]["random_seed"])

        self.assertIsNotNone(env.np_random_seed)

    def test_a_seeded_reset_still_uses_the_seed_it_was_given(self):
        """The control. Answering every reset with a fresh generator would pass
        the test above and lose the reproducibility the competition rests on."""
        parameters = _parameters(0)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        self.addCleanup(env.close)

        env.reset(seed=parameters["scenario"]["random_seed"])

        self.assertEqual(env.np_random_seed, 0)

    def test_two_unseeded_resets_do_not_land_on_the_same_world(self):
        """What ``null`` is for. Both releases are shuffled from the generator,
        so identical orders would mean it was never drawn."""
        orders = []
        for _ in range(2):
            env = BalloonPoppingEnv(render_mode=None, parameters=_parameters(None))
            self.addCleanup(env.close)
            env.reset(seed=None)
            orders.append(tuple(env._balloon_release_at_step.tolist()))

        self.assertNotEqual(orders[0], orders[1])


if __name__ == "__main__":
    unittest.main()
