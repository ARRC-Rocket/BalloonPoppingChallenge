"""Return-contract test for `evaluate_scenario`.

`evaluate_scenario` is the package's public evaluation entry point. Its
diagnostics now go through `logging`, which is silent unless the caller
configures it, so library callers (the Colab example, `run_env_agent.py`)
need the run's result as a return value. This test pins that contract.

Runtime test: importing `evaluate` pulls `balloon_world` and the simulation
stack, so it skips when that stack is absent. The environment is mocked, so
no real flight simulation runs.
"""

import importlib.util
import unittest
from unittest.mock import patch


def _simulation_stack_installed():
    """True when the heavy simulation stack (rocketpy) is installed."""
    return importlib.util.find_spec("rocketpy") is not None


@unittest.skipUnless(_simulation_stack_installed(), "simulation stack not installed")
class TestEvaluateScenarioReturn(unittest.TestCase):
    """evaluate_scenario must return the popped-balloon count to its caller."""

    def test_returns_popped_count(self):
        from BalloonPoppingGymEnv.evaluation.evaluate import evaluate_scenario

        with patch(
            "BalloonPoppingGymEnv.evaluation.evaluate.BalloonPoppingEnv"
        ) as mock_env_cls:
            env = mock_env_cls.return_value
            env.reset.return_value = ({}, {})
            env.step.return_value = ({}, 0.0, True, False, {"popped_count": 7})

            class _FakeAgent:
                def __init__(self, *args, **kwargs):
                    pass

                def get_action(self, observation):
                    return {}

            result = evaluate_scenario(_FakeAgent, scenario_number=0)

        self.assertEqual(result, 7)


if __name__ == "__main__":
    unittest.main()
