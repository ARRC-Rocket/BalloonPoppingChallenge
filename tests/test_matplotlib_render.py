"""The default renderer, on the frames drawn before the rocket has a state.

Issue #89: with Matplotlib 3.11 the run dies on the first frame, because
``Line3D.draw`` reads ``_verts3d[0].shape`` when a coordinate is invalid for the
axis scale, and the rocket state is all-NaN until the launch action builds the
flight. Passing lists made those entries lists.

These assert the types rather than only that a frame draws. The crashing branch
does not exist on Matplotlib 3.10, so a draw-only test passes there whether or
not the fix is applied, and the project's own CI pins 3.10 through the lockfile.
Asserting the entries are arrays pins the fix on every version, and is what the
fix actually guarantees.
"""

import unittest
from importlib.util import find_spec
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_0 = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_0_parameters.yaml"
)

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheMatplotlibRendererBeforeLaunch(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        self.addCleanup(plt.close, "all")
        parameters = yaml.safe_load(SCENARIO_0.read_text(encoding="utf-8-sig"))
        self.env = BalloonPoppingEnv(render_mode="matplotlib", parameters=parameters)
        self.env.reset(seed=parameters["scenario"]["random_seed"])

    def test_the_rocket_state_really_is_nan_before_launch(self):
        """The precondition, so this suite cannot stop reaching the branch.

        If the environment ever started the rocket at a finite position, the
        tests below would keep passing while covering nothing.
        """
        self.assertTrue(
            np.isnan(np.asarray(self.env._rocket_states, dtype=float)).all()
        )

    def test_a_frame_before_launch_leaves_the_line_holding_arrays(self):
        self.env._render_frame()

        verts = self.env.render_rocket[0].get_data_3d()
        for axis, values in zip("xyz", verts, strict=True):
            with self.subTest(axis=axis):
                self.assertIsInstance(
                    values,
                    np.ndarray,
                    "Line3D.draw reads .shape off these when a coordinate is "
                    "invalid for the axis scale, which every pre-launch frame is",
                )

    def test_a_frame_before_launch_draws(self):
        """Draws rather than raising, which is the user-visible half.

        Passes on Matplotlib 3.10 with or without the fix, so it is here for the
        version that does crash rather than as the pin.
        """
        self.env._render_frame()
        self.env._render_frame()

        self.env.render_canvas.get_figure().canvas.draw()


if __name__ == "__main__":
    unittest.main()
