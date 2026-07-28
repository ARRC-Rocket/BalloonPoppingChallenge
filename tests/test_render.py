"""Behavioural test for issue #26: the vpython renderer must draw every balloon.

`_render_frame`'s vpython branch previously created a single `sphere` and
positioned it from `_balloon_states[0]`, so only the first balloon showed.
This test mocks the `vpython` module and checks that a reset creates one sphere
per balloon **and gives each one its own position**. Counting spheres alone is not
enough: a renderer that builds N of them and assigns `.pos` on only the first
would still be the #26 bug, and #40 removed the assertion that indirectly caught
that. Asserting positions is also stable against unrelated canvas or rocket-arrow
changes, which is what made the old `vector()` call count brittle in the first
place.

Runtime test: needs the simulation stack to build the env. The `vpython`
package itself is mocked, so it does not need to be installed.
"""

import importlib.util
import sys
from dataclasses import dataclass
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
@dataclass(frozen=True)
class FakeVector:
    """Stands in for ``vpython.vector`` while keeping its identity.

    The earlier stand-in was ``lambda x, y, z: (x, y, z)``, which made the
    renderer's ``balloon.pos = vector(...)`` indistinguishable from
    ``balloon.pos = (x, y, z)``. Real VPython is not indifferent: assigning to
    ``pos`` hands the value to a vector's own setter, which reads ``_x``, ``_y``
    and ``_z`` off it, so a plain tuple fails in the renderer while passing
    every assertion here.

    Comparable and iterable so the assertions stay about coordinates, distinct
    from a tuple so bypassing ``vector()`` cannot go unnoticed. That is what the
    call-count assertion #40 deleted was reaching for, without the fragility of
    counting every call in the frame.
    """

    x: object
    y: object
    z: object

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


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
        # One distinct mock per sphere, so each balloon's assignments are
        # attributable, and a vector() that keeps its own type.
        spheres = [MagicMock(name=f"sphere{index}") for index in range(num)]
        fake_vpython.sphere.side_effect = spheres
        fake_vpython.vector.side_effect = FakeVector
        with patch.dict(sys.modules, {"vpython": fake_vpython}):
            env.reset(seed=0)

        self.assertEqual(
            fake_vpython.sphere.call_count,
            num,
            "vpython render must create one sphere per balloon",
        )
        self.assertEqual(len(env.render_balloons), num)

        # The half #40 dropped: every balloon has to be placed, at its own
        # position. Creating N spheres and moving only the first is the #26 bug.
        for index, (drawn, state) in enumerate(
            zip(spheres, env._balloon_states, strict=True)
        ):
            with self.subTest(balloon=index):
                # The type as well as the coordinates: assigning the tuple
                # directly, without going through vector(), places every balloon
                # correctly here and breaks the real renderer.
                self.assertIsInstance(
                    drawn.pos, FakeVector, "balloon position bypassed vector()"
                )
                self.assertEqual(
                    drawn.pos,
                    FakeVector(state[0], state[1], state[2]),
                    "balloon was not drawn at its own position",
                )

        # A fixture check, not a renderer one. The per-balloon equality above
        # already fails if every sphere sits at balloon 0, and the renderer owes
        # nothing here: one balloon, or two that happen to coincide, would both
        # be legitimate. This says the scenario still spreads them out, so the
        # comparison above is testing something.
        drawn_positions = {tuple(drawn.pos) for drawn in spheres}
        self.assertGreater(
            len(drawn_positions),
            1,
            "the scenario no longer spreads its balloons out, so this fixture "
            "cannot distinguish per-balloon placement from a shared position",
        )

    def test_the_coordinates_are_not_transposed(self):
        """Scenario 0's column sits at x = y = 0, which hides an axis swap.

        Every balloon there has the same x and y, so drawing ``(y, x, z)`` is
        indistinguishable from ``(x, y, z)`` and the assertions above pass. This
        feeds in asymmetric positions so the ordering actually matters.
        """
        import yaml

        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

        with open(SCENARIO_0_PARAMS, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        num = params["balloon"]["num"]

        env = BalloonPoppingEnv(render_mode="vpython", parameters=params)
        fake_vpython = MagicMock()
        spheres = [MagicMock(name=f"sphere{index}") for index in range(num)]
        fake_vpython.sphere.side_effect = spheres
        fake_vpython.vector.side_effect = FakeVector

        # Not a shared linear ramp: three axes stepping together would let a
        # swap that also rescaled go unnoticed. Mixed signs and a quadratic.
        asymmetric = [
            (1.25 + index, -7.0 - 3.0 * index, 200.0 + index**2) for index in range(num)
        ]
        with patch.dict(sys.modules, {"vpython": fake_vpython}):
            env.reset(seed=0)
            for index, position in enumerate(asymmetric):
                env._balloon_states[index, :3] = position
            env._render_frame()

        for index, (drawn, position) in enumerate(
            zip(spheres, asymmetric, strict=True)
        ):
            with self.subTest(balloon=index):
                self.assertIsInstance(drawn.pos, FakeVector)
                self.assertEqual(
                    drawn.pos, FakeVector(*position), "coordinates were reordered"
                )


if __name__ == "__main__":
    unittest.main()
