"""Behavioural test for issue #2: Monte Carlo output must not land in the package dir.

`MonteCarlo` writes `.inputs/.outputs/.errors.txt` next to its `filename`. The
balloon env only consumes the in-memory results, so that path must be a
writable temp location, not the installed package directory (which can be
read-only, e.g. a site-packages install).

Runtime test: needs the simulation stack. Skips cleanly when it is absent.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DATA_DIR = REPO_ROOT / "BalloonPoppingGymEnv" / "envs" / "data"
SCENARIO_1_PARAMS = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_1_parameters.yaml"
)

try:
    import yaml

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

    _STACK_AVAILABLE = True
except ImportError:
    _STACK_AVAILABLE = False

_MONTE_CARLO = "BalloonPoppingGymEnv.envs.balloon_world.MonteCarlo"


class _StopBeforeSimulation(Exception):
    """Sentinel: aborts the run once MonteCarlo is constructed."""


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestMonteCarloOutputPath(unittest.TestCase):
    """Issue #2: the Monte Carlo output path must be outside the package directory."""

    def test_monte_carlo_filename_is_not_in_package_dir(self):
        with open(SCENARIO_1_PARAMS, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        env = BalloonPoppingEnv(render_mode=None, parameters=params)

        # Capture the filename MonteCarlo is constructed with, then abort the run.
        with patch(_MONTE_CARLO, side_effect=_StopBeforeSimulation) as mc:
            try:
                env.reset(seed=0)
            except _StopBeforeSimulation:
                pass

        self.assertTrue(mc.called, "MonteCarlo was not invoked for scenario 1")
        filename = Path(str(mc.call_args.kwargs["filename"])).resolve()
        package_data = str(PACKAGE_DATA_DIR.resolve())
        self.assertFalse(
            str(filename).startswith(package_data),
            f"MonteCarlo writes into the package directory: {filename}",
        )
        # Per-process name: concurrent processes / users must not collide on
        # one fixed path in the shared system temp directory.
        self.assertIn(
            str(os.getpid()),
            filename.name,
            f"Monte Carlo output filename should be per-process unique: {filename}",
        )


if __name__ == "__main__":
    unittest.main()
