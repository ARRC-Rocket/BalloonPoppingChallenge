"""Behavioural test for issue #2: Monte Carlo output must not land in the package dir.

`MonteCarlo` writes `.inputs/.outputs/.errors.txt` next to its `filename`. The
balloon env only consumes the in-memory results, so that path must be a
writable temp location, not the installed package directory (which can be
read-only, e.g. a site-packages install).

Runtime test: needs the simulation stack. Skips cleanly when it is absent.
"""

import os
import tempfile
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

# Same gate as the golden masters: the cleanup check needs the real Monte Carlo.
_RUN_SLOW = os.environ.get("BPC_RUN_SLOW_TESTS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)


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
        self.assertTrue(
            str(filename).startswith(str(Path(tempfile.gettempdir()).resolve())),
            f"Monte Carlo output should be under the system temp dir: {filename}",
        )

    def test_two_runs_do_not_share_an_output_path(self):
        """Concurrent runs must not write over each other.

        This used to be asserted as "the PID is in the filename", which is one
        way to get the property and not the property itself. Comparing two runs
        covers what actually matters, and it holds for the directory mkdtemp now
        makes as well as it did for the old name.
        """
        paths = []
        for _ in range(2):
            with open(SCENARIO_1_PARAMS, "r", encoding="utf-8") as f:
                params = yaml.safe_load(f)
            env = BalloonPoppingEnv(render_mode=None, parameters=params)
            with patch(_MONTE_CARLO, side_effect=_StopBeforeSimulation) as mc:
                try:
                    env.reset(seed=0)
                except _StopBeforeSimulation:
                    pass
            paths.append(Path(str(mc.call_args.kwargs["filename"])).resolve())

        self.assertNotEqual(paths[0], paths[1])

    @unittest.skipUnless(_RUN_SLOW, "slow: runs the scenario 1 Monte Carlo")
    def test_the_output_files_do_not_survive_the_reset(self):
        """Issue #91, which is what a competitor actually feels.

        MonteCarlo writes .inputs.txt, .outputs.txt and .errors.txt beside its
        filename, and for scenario 1 the outputs file is 169 MB. Nothing reads
        them back. Left behind they accumulate one set per run, which fills the
        shared temp directory within tens of runs and then breaks runs in ways
        that do not point back here. Measured before the fix: 22 files, 3.7 GB.

        Asserted over the whole temp directory rather than the three known
        suffixes, so a fourth file would fail this too. The real Monte Carlo has
        to run: aborting at construction would leave nothing to clean up and the
        test would pass against no cleanup at all.
        """
        temp_dir = Path(tempfile.gettempdir())
        before = set(temp_dir.iterdir())

        with open(SCENARIO_1_PARAMS, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        env = BalloonPoppingEnv(render_mode=None, parameters=params)
        env.reset(seed=params["scenario"]["random_seed"])

        # The precondition: the run has to have produced the arrays, or there
        # was no Monte Carlo and nothing to leave behind.
        self.assertEqual(env._balloon_flights.shape[0], params["balloon"]["num"])

        left_behind = set(temp_dir.iterdir()) - before
        self.assertEqual(
            sorted(str(path) for path in left_behind),
            [],
            "the Monte Carlo output was left in the shared temp directory",
        )


if __name__ == "__main__":
    unittest.main()
