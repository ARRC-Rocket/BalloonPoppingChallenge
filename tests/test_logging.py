"""Logging tests: diagnostic prints are routed through the standard logging.

The diagnostic ``print()`` calls in the environment and the evaluation entry
point were converted to module-level loggers (no behaviour change beyond the
output sink). These tests pin that behaviour so the prints do not creep back.

``test_close_emits_through_module_logger`` builds the env with ``__new__``
because ``close()`` reads no instance attributes. Both tests still import the
modules under test, which pull in the simulation stack, so they are skipped
when that stack is not installed (mirroring the rest of the suite).
"""

import importlib.util
import logging
import unittest


def _simulation_stack_installed():
    """True when rocketpy is installed (probed via find_spec, without importing it)."""
    return importlib.util.find_spec("rocketpy") is not None


@unittest.skipUnless(_simulation_stack_installed(), "simulation stack not installed")
class TestLogging(unittest.TestCase):
    """Diagnostic prints must be emitted through module-level loggers."""

    def test_close_emits_through_module_logger(self):
        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

        # close() touches no instance state, so __new__ avoids building the env.
        env = BalloonPoppingEnv.__new__(BalloonPoppingEnv)
        with self.assertLogs(
            "BalloonPoppingGymEnv.envs.balloon_world", level="DEBUG"
        ) as captured:
            env.close()

        self.assertTrue(
            any("closing environment" in message for message in captured.output),
            "close() must log 'closing environment' via the module logger",
        )

    def test_modules_define_named_loggers(self):
        from BalloonPoppingGymEnv.envs import balloon_world
        from BalloonPoppingGymEnv.evaluation import evaluate

        for module, expected_name in (
            (balloon_world, "BalloonPoppingGymEnv.envs.balloon_world"),
            (evaluate, "BalloonPoppingGymEnv.evaluation.evaluate"),
        ):
            self.assertIsInstance(module.logger, logging.Logger)
            self.assertEqual(module.logger.name, expected_name)


if __name__ == "__main__":
    unittest.main()
