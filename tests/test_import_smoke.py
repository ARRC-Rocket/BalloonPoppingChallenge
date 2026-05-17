"""Import smoke test (issue #21).

Cheapest signal that an install is functional: the gymnasium environment
package imports and registers, and the example agents import.

`test_import_package` is light -- it only needs `gymnasium`, a hard runtime
dependency. `test_import_example_agents` pulls in `balloon_world` and the
full simulation stack (rocketpy, via the ActiveRocketPy submodule), so it is
skipped when that stack is absent. The skip guard probes `rocketpy` only --
a broken `BalloonPoppingGymEnv` still fails loudly inside the test body.
"""

import importlib
import unittest


def _simulation_stack_installed():
    """True when the heavy simulation stack (rocketpy) is importable."""
    try:
        importlib.import_module("rocketpy")
    except ImportError:
        return False
    return True


class TestImportSmoke(unittest.TestCase):
    """Issue #21: the package and example agents must import successfully."""

    def test_import_package(self):
        """`import BalloonPoppingGymEnv` registers the env without the sim stack."""
        package = importlib.import_module("BalloonPoppingGymEnv")
        self.assertIsNotNone(package)

    @unittest.skipUnless(
        _simulation_stack_installed(), "simulation stack not installed"
    )
    def test_import_example_agents(self):
        """The example agents and their base class import cleanly."""
        importlib.import_module("BalloonPoppingGymEnv.agents.base_agent")
        agents = importlib.import_module("BalloonPoppingGymEnv.agents.example_agents")
        for name in (
            "SineCommandAgent",
            "AttitudeRateControlAgent",
            "NavigationAgent",
        ):
            self.assertTrue(hasattr(agents, name), f"example agent {name} is missing")


if __name__ == "__main__":
    unittest.main()
