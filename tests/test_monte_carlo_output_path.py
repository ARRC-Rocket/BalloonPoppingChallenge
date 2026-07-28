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
from importlib.util import find_spec
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

# ``find_spec`` answers "is the package installed", which is the only case that
# justifies a skip. The import itself stays outside any guard: ``import rocketpy``
# runs the package's own ``__init__``, so an ImportError raised there means an
# installed but broken stack, which is exactly what these tests exist to catch and
# must fail rather than skip.
_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import yaml

    from rocketpy import MonteCarlo

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv

_MONTE_CARLO = "BalloonPoppingGymEnv.envs.balloon_world.MonteCarlo"

# The prefix production uses, so the helper below can tell our workspace from
# any other mkdtemp caller it happens to intercept.
_WORKSPACE_PREFIX = "balloon_sim_"

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
        # is_relative_to rather than startswith: "/tmp/data_other/x" starts with
        # "/tmp/data" and is not inside it.
        self.assertFalse(
            filename.is_relative_to(PACKAGE_DATA_DIR.resolve()),
            f"MonteCarlo writes into the package directory: {filename}",
        )
        self.assertTrue(
            filename.is_relative_to(Path(tempfile.gettempdir()).resolve()),
            f"Monte Carlo output should be under the system temp dir: {filename}",
        )

    def test_separate_resets_use_distinct_output_paths(self):
        """Two resets do not write over each other.

        Named for what it does rather than for concurrency: it runs them one
        after another, so it shows distinct paths rather than proving anything
        about two live invocations. mkdtemp is what provides the exclusive
        creation, and testing the standard library is not this file's job.

        This used to be asserted as "the PID is in the filename", which is one
        way to get the property and not the property itself.
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

    def _one_balloon_scenario_1(self):
        """Scenario 1 cut to a single balloon.

        The file writing is what these tests are about and one iteration
        exercises all of it: MonteCarlo sets up the three files, writes to them
        and reads the results back regardless of how many simulations run. A
        hundred balloons would add ten seconds and another 169 MB to say the
        same thing, and #49 already pays for a full scenario-1 run.
        """
        with open(SCENARIO_1_PARAMS, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        params["balloon"]["num"] = 1
        return params

    def _reset_recording_workspaces(self, params, own_directory):
        """Reset the environment, returning the workspaces it created.

        Redirected into a directory this test owns so the assertions are about
        the files this reset made. Watching the shared system temp directory
        instead would fail whenever anything else on the machine wrote there
        during the run, which on CI is not a hypothetical.
        """
        created = []
        real_mkdtemp = tempfile.mkdtemp

        def recording_mkdtemp(*args, **kwargs):
            # balloon_world does ``import tempfile``, so its ``tempfile`` is the
            # module itself and patching through it patches every caller in the
            # process. Anything else that reaches mkdtemp during the run,
            # including TemporaryDirectory, would otherwise land in this
            # directory and be counted as a workspace. Only ours is taken.
            if kwargs.get("prefix") != _WORKSPACE_PREFIX:
                return real_mkdtemp(*args, **kwargs)
            made = real_mkdtemp(prefix=_WORKSPACE_PREFIX, dir=own_directory)
            created.append(Path(made))
            return made

        env = BalloonPoppingEnv(render_mode=None, parameters=params)
        with patch(
            "BalloonPoppingGymEnv.envs.balloon_world.tempfile.mkdtemp",
            side_effect=recording_mkdtemp,
        ):
            env.reset(seed=params["scenario"]["random_seed"])
        return env, created

    @unittest.skipUnless(_RUN_SLOW, "slow: runs the scenario 1 Monte Carlo")
    def test_the_output_files_do_not_survive_a_successful_reset(self):
        """Issue #91, which is what a competitor actually feels.

        MonteCarlo writes .inputs.txt, .outputs.txt and .errors.txt beside its
        filename, and at a hundred balloons the outputs file is 169 MB. A
        successful reset reads none of them back. Left behind they accumulate
        one set per run, which fills the shared temp directory within tens of
        runs and then breaks runs in ways that do not point back here. Measured
        before the fix: 22 files, 3.7 GB.

        The real Monte Carlo has to run. Aborting at construction would leave
        nothing to clean up and the test would pass against no cleanup at all.
        """
        params = self._one_balloon_scenario_1()
        with tempfile.TemporaryDirectory() as own_directory:
            env, created = self._reset_recording_workspaces(params, own_directory)

            # Preconditions, or the assertions below are about nothing: the run
            # produced its arrays, and it did so through a workspace.
            self.assertEqual(env._balloon_flights.shape[0], params["balloon"]["num"])
            self.assertEqual(len(created), 1, f"workspaces created: {created}")

            self.assertFalse(
                created[0].exists(),
                f"the Monte Carlo output was left at {created[0]}",
            )
            self.assertEqual(
                sorted(path.name for path in Path(own_directory).iterdir()),
                [],
                "something was left beside the workspace",
            )

    @unittest.skipUnless(_RUN_SLOW, "slow: runs the scenario 1 Monte Carlo")
    def test_a_failed_reset_keeps_the_workspace(self):
        """The other half, which cleaning up unconditionally would take away.

        RocketPy keeps the failing stochastic inputs on purpose: a serial
        simulation error appends them to .errors.txt before re-raising, and
        Ctrl-C prints "Files saved." Deleting the directory while the exception
        unwinds makes that message a lie and leaves whoever has to explain the
        failure with nothing.

        The failure is injected after ``simulate`` rather than inside it, so it
        also covers the conversion below: an incomplete result is exactly when
        the inputs that produced it are worth having.
        """
        params = self._one_balloon_scenario_1()
        with tempfile.TemporaryDirectory() as own_directory:
            with patch(
                "BalloonPoppingGymEnv.envs.balloon_world.pm.geodetic2enu",
                side_effect=RuntimeError("conversion blew up"),
            ):
                with self.assertRaises(RuntimeError):
                    self._reset_recording_workspaces(params, own_directory)

            left = sorted(Path(own_directory).iterdir())
            self.assertEqual(len(left), 1, f"expected the workspace to survive: {left}")
            self.assertTrue(
                any(path.name.endswith(".errors.txt") for path in left[0].iterdir()),
                f"the error log RocketPy wrote is gone: {list(left[0].iterdir())}",
            )

    @unittest.skipUnless(_RUN_SLOW, "slow: runs the scenario 1 Monte Carlo")
    def test_a_failure_inside_the_simulation_keeps_what_it_wrote(self):
        """The case the retention is actually for, rather than one next to it.

        The test above fails during the conversion, by which point the files are
        already closed. This one fails the way RocketPy does: append the failing
        stochastic inputs to the error file, then raise. Verified against the
        pinned submodule, monte_carlo.py lines 317 to 321.

        Asserts the content survives, not only the file. The file exists either
        way because setup creates all three empty.
        """
        params = self._one_balloon_scenario_1()
        marker = "the inputs that failed"
        real_simulate = MonteCarlo.simulate

        def simulate_then_fail(self, *args, **kwargs):
            with open(self._error_file, "a", encoding="utf-8") as handle:
                handle.write(marker)
            raise RuntimeError("simulation blew up")

        with tempfile.TemporaryDirectory() as own_directory:
            with patch.object(MonteCarlo, "simulate", simulate_then_fail):
                with self.assertRaises(RuntimeError):
                    self._reset_recording_workspaces(params, own_directory)

            left = sorted(Path(own_directory).iterdir())
            self.assertEqual(len(left), 1, f"expected the workspace to survive: {left}")
            errors = [
                path for path in left[0].iterdir() if path.name.endswith(".errors.txt")
            ]
            self.assertEqual(len(errors), 1, f"no error log: {list(left[0].iterdir())}")
            self.assertIn(marker, errors[0].read_text(encoding="utf-8"))
        self.assertIs(MonteCarlo.simulate, real_simulate)

    @unittest.skipUnless(_RUN_SLOW, "slow: runs the scenario 1 Monte Carlo")
    def test_a_failure_before_anything_is_written_leaves_nothing(self):
        """An empty workspace is not a diagnostic, it is the leak again.

        Every failure inside the MonteCarlo constructor happens before a file
        exists. Keeping the directory for those would put one back on every
        attempt, which is what this change set out to stop. Measured against the
        first version of this fix: one empty directory per failed construction.
        """
        params = self._one_balloon_scenario_1()
        with tempfile.TemporaryDirectory() as own_directory:
            with patch(_MONTE_CARLO, side_effect=RuntimeError("constructor blew up")):
                with self.assertRaises(RuntimeError):
                    self._reset_recording_workspaces(params, own_directory)

            self.assertEqual(
                sorted(path.name for path in Path(own_directory).iterdir()),
                [],
                "an empty workspace was kept",
            )


if __name__ == "__main__":
    unittest.main()
