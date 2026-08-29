"""An ignored action field has to say why it was ignored.

`_usable_action_fields` catches every `Exception`, which is deliberate: a torch
tensor still carrying its graph raises `RuntimeError`, and an action that is not
a mapping raises `TypeError` on the lookup, so narrowing the list is how fields
go missing instead. The cost is that a defect in the conversion reads exactly
like a competitor sending a string, and the log line said neither.

Only ``rocketpy`` is guarded: a missing simulation stack is a legitimate skip,
but a broken import inside this package is a failure and must stay loud.
"""

import logging
import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import (
        _usable_action_fields,
        check_action,
    )


def _action(**overrides):
    action = {
        "launch": np.array(0.0),
        "launch_inclination_heading": np.zeros(2),
        "tvc": np.zeros(2),
        "throttle": np.array(0.0),
        "roll": np.array(0.0),
    }
    action.update(overrides)
    return action


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheReasonIsCarried(unittest.TestCase):
    def test_a_raised_exception_keeps_its_type(self):
        """The point of the change. A NameError from our own conversion has to
        be distinguishable from a competitor's unconvertible value."""
        _, unusable = _usable_action_fields(_action(tvc="not a command"))

        self.assertIn("tvc", unusable)
        self.assertIn("ValueError", unusable["tvc"])

    def test_the_wrong_count_says_what_was_expected(self):
        _, unusable = _usable_action_fields(_action(tvc=np.zeros(3)))

        self.assertEqual(unusable["tvc"], "expected 2 values, got 3")

    def test_a_non_finite_value_says_so(self):
        _, unusable = _usable_action_fields(_action(throttle=np.array(np.nan)))

        self.assertEqual(unusable["throttle"], "not every value is finite")

    def test_a_missing_field_keeps_its_lookup_error(self):
        action = _action()
        del action["roll"]

        _, unusable = _usable_action_fields(action)

        self.assertIn("KeyError", unusable["roll"])

    def test_an_unexpected_exception_type_survives(self):
        """Anything at all, which is what the broad catch is there for."""

        class Hostile:
            def __array__(self, *_args, **_kwargs):
                raise RuntimeError("still attached to a graph")

        _, unusable = _usable_action_fields(_action(tvc=Hostile()))

        self.assertIn("RuntimeError", unusable["tvc"])
        self.assertIn("still attached to a graph", unusable["tvc"])


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestThePublicAnswerIsUnchanged(unittest.TestCase):
    """`check_action` is what an agent calls, so it still answers with names."""

    def test_it_is_a_sorted_list_of_names(self):
        result = check_action(_action(tvc="x", roll=np.array(np.inf)))

        self.assertEqual(result, ["roll", "tvc"])

    def test_a_usable_action_is_an_empty_list(self):
        self.assertEqual(check_action(_action()), [])


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheReasonReachesTheLog(unittest.TestCase):
    def test_the_warning_names_the_reason(self):
        """Through the environment, since a reason nobody can read is no use."""
        import copy

        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        loaded = load_scenario_parameters(0)
        parameters = copy.deepcopy(loaded[0] if isinstance(loaded, tuple) else loaded)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        self.addCleanup(env.close)
        env.reset(seed=0)

        with self.assertLogs(
            "BalloonPoppingGymEnv.envs.balloon_world", level=logging.WARNING
        ) as captured:
            env.step(_action(tvc="not a command"))

        self.assertTrue(
            any("ValueError" in line for line in captured.output),
            captured.output,
        )


if __name__ == "__main__":
    unittest.main()
