"""Behavioural test for issue #26: the vpython renderer must draw every balloon.

`_render_frame`'s vpython branch previously created a single `sphere` and
positioned it from `_balloon_states[0]`, so only the first balloon showed.
This test mocks the `vpython` module and checks that a reset creates one
sphere per balloon.

Runtime test: needs the simulation stack to build the env. The `vpython`
package itself is mocked, so it does not need to be installed.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_0_PARAMS = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_0_parameters.yaml"
)


def _simulation_stack_installed():
    """True when the heavy simulation stack (rocketpy) is installed."""
    return importlib.util.find_spec("rocketpy") is not None


@unittest.skipUnless(_simulation_stack_installed(), "simulation stack not installed")
class TestVpythonRendersAllBalloons(unittest.TestCase):
    """Issue #26: the vpython renderer must create one sphere per balloon."""

    def test_reset_creates_one_sphere_per_balloon(self):
        import yaml

        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

        with open(SCENARIO_0_PARAMS, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        num = params["balloon"]["num"]

        env = BalloonPoppingEnv(render_mode="vpython", parameters=params)
        fake_vpython = MagicMock()
        with patch.dict(sys.modules, {"vpython": fake_vpython}):
            env.reset(seed=0)

        self.assertEqual(
            fake_vpython.sphere.call_count,
            num,
            "vpython render must create one sphere per balloon",
        )
        self.assertEqual(len(env.render_balloons), num)
        # vector(): 1 for the canvas centre + 1 per balloon in the update loop
        self.assertEqual(fake_vpython.vector.call_count, num + 1)


if __name__ == "__main__":
    unittest.main()
