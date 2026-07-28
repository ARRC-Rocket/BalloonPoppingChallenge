"""The default renderer, on the frames drawn before the rocket has a state.

Issue #89: with Matplotlib 3.11 the run dies on the first frame, because
``Line3D.draw`` reads ``_verts3d[0].shape`` when a coordinate is invalid for the
axis scale, and the rocket state is all-NaN until the launch action builds the
flight. Passing lists made those entries lists.

Both halves are load-bearing, in opposite environments.

CI installs with ``pip install -r requirements-dev.txt``, and the only
constraint anywhere is ActiveRocketPy's ``matplotlib>=3.9.0``, with no upper
bound. So CI resolves whatever is current, which is how it picked up 3.11.1 and
went red. There, rendering a frame is enough to catch this.

``uv sync`` resolves 3.10.9 from the lockfile, and that is what a local run
gets. The crashing branch does not exist there, so the draw test passes with or
without the fix. What holds on 3.10 is the assertion about what the line is
holding, which is what the fix actually guarantees.

Neither environment covers this on its own, so both assertions stay.
"""

import unittest
from importlib.util import find_spec
from pathlib import Path

import matplotlib
import numpy as np
import yaml

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

        # Scoped to this test, not set at import. matplotlib.use at module level
        # changes the backend for the whole pytest session, and this file has no
        # business deciding that for tests that never render.
        previous_backend = matplotlib.get_backend()
        self.addCleanup(plt.switch_backend, previous_backend)
        plt.switch_backend("Agg")
        self.addCleanup(plt.close, "all")
        parameters = yaml.safe_load(SCENARIO_0.read_text(encoding="utf-8-sig"))
        self.env = BalloonPoppingEnv(render_mode="matplotlib", parameters=parameters)
        _observation, self.info = self.env.reset(
            seed=parameters["scenario"]["random_seed"]
        )

    def test_the_rocket_state_really_is_nan_before_launch(self):
        """The precondition, so this suite cannot stop reaching the branch.

        If the environment ever started the rocket at a finite position, the
        tests below would keep passing while covering nothing.

        Read from the info the environment hands back rather than from
        ``_rocket_states``, so this is about the state the environment reports.
        """
        self.assertTrue(
            np.isnan(np.asarray(self.info["rocket_states"], dtype=float)).all()
        )

    def test_a_frame_before_launch_leaves_the_line_holding_arrays(self):
        self.env._render_frame()

        # get_data_3d is documented public API. What is asserted is the property
        # Line3D.draw actually requires, rather than a specific numpy type:
        # `_verts3d[0].shape` is read when a coordinate is invalid for the axis
        # scale, which every pre-launch frame is. Pinning ndarray instead would
        # over-specify a storage detail matplotlib does not promise.
        verts = self.env.render_rocket[0].get_data_3d()
        for axis, values in zip("xyz", verts, strict=True):
            with self.subTest(axis=axis):
                self.assertTrue(
                    hasattr(values, "shape"),
                    f"Line3D.draw reads .shape off this; got {type(values).__name__}",
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
