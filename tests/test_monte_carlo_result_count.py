"""A short Monte Carlo result set must not become a world.

``MonteCarlo.simulate()`` does not always raise when it stops early. The pinned
RocketPy catches ``KeyboardInterrupt``, appends to its error file, prints
"Keyboard interrupt received. Files saved." and returns normally; only
``Exception`` is re-raised. So a Ctrl-C part way through a hundred balloons
comes back as fewer trajectories rather than as a failure.

One returned trajectory is the case that matters. The release shift in
``__generate_balloon_flights`` indexes with ``arange(num_balloons)`` against one
release step per *requested* balloon, so a ``(1, 6, T)`` array broadcasts to a
full ``(num, 6, T)`` one in which every balloon flies the same path. The episode
then scores normally against a world that was never simulated.

These patch ``MonteCarlo`` rather than running one, so they cost an environment
build and no simulation.
"""

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_1_PARAMS = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_1_parameters.yaml"
)

# ``find_spec`` answers "is the package installed", which is the only case that
# justifies a skip. The imports stay outside the guard: an ImportError from them
# means an installed but broken stack, which must fail rather than skip.
_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import yaml

    from BalloonPoppingGymEnv.envs.balloon_world import (
        BalloonPoppingEnv,
        _monte_carlo_workspace_has_files,
    )

_MONTE_CARLO = "BalloonPoppingGymEnv.envs.balloon_world.MonteCarlo"

# Small enough that the assertions read clearly. The count is what is under
# test, so the number of balloons is not.
BALLOONS = 4
TIMESTEPS = 40


def _results_for(count):
    """A results dict shaped the way ``simulate()`` returns one, with ``count`` rows."""
    return {
        "x": [np.linspace(0.0, 10.0 + index, TIMESTEPS) for index in range(count)],
        "y": [np.linspace(0.0, 20.0 + index, TIMESTEPS) for index in range(count)],
        "z": [np.linspace(0.0, 30.0 + index, TIMESTEPS) for index in range(count)],
        "vx": [np.full(TIMESTEPS, 1.0 + index) for index in range(count)],
        "vy": [np.full(TIMESTEPS, 2.0 + index) for index in range(count)],
        "vz": [np.full(TIMESTEPS, 3.0 + index) for index in range(count)],
        "lat0": [24.0 + 0.001 * index for index in range(count)],
        "lon0": [121.0 + 0.001 * index for index in range(count)],
    }


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheBroadcastThisGuards(unittest.TestCase):
    """Why one result is worse than none, asserted rather than described.

    Without this test the guard's own tests would pass just as well if the
    indexing below had never had the problem, and there would be nothing to say
    the guard is still load bearing.
    """

    def test_one_trajectory_broadcasts_up_to_the_full_balloon_count(self):
        flights = np.arange(1 * 6 * TIMESTEPS, dtype=float).reshape(1, 6, TIMESTEPS)
        release_steps = np.arange(BALLOONS)

        # The expression from __generate_balloon_flights, on a short array.
        source_idx = np.clip(
            np.arange(TIMESTEPS)[None, :] - release_steps[:, None], 0, TIMESTEPS - 1
        )
        shifted = flights[
            np.arange(flights.shape[0])[:, None, None],
            np.arange(6)[None, :, None],
            source_idx[:, None, :],
        ]

        # A full-sized world out of one simulated balloon, and no error.
        self.assertEqual(shifted.shape, (BALLOONS, 6, TIMESTEPS))
        for balloon in range(1, BALLOONS):
            # Every one of them is the same trajectory, only shifted in time.
            np.testing.assert_array_equal(
                shifted[balloon, :, balloon:], shifted[0, :, : TIMESTEPS - balloon]
            )


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestAShortResultSetIsRefused(unittest.TestCase):
    def _reset_with(self, results):
        with open(SCENARIO_1_PARAMS, "r", encoding="utf-8") as parameter_file:
            params = yaml.safe_load(parameter_file)
        params["balloon"]["num"] = BALLOONS
        env = BalloonPoppingEnv(render_mode=None, parameters=params)
        monte_carlo = MagicMock()
        monte_carlo.return_value.simulate.return_value = results
        with patch(_MONTE_CARLO, monte_carlo):
            env.reset(seed=0)
        return env

    def test_a_single_trajectory_is_refused(self):
        """The silent case: this is the one that used to build a whole world."""
        with self.assertRaises(RuntimeError) as caught:
            self._reset_with(_results_for(1))

        message = str(caught.exception)
        self.assertIn(str(BALLOONS), message)
        self.assertIn("interrupt", message)

    def test_a_partial_result_set_is_refused(self):
        with self.assertRaises(RuntimeError):
            self._reset_with(_results_for(BALLOONS - 1))

    def test_no_results_at_all_are_refused(self):
        with self.assertRaises(RuntimeError):
            self._reset_with(_results_for(0))

    def test_a_missing_key_is_refused_rather_than_raising_a_key_error(self):
        results = _results_for(BALLOONS)
        del results["vz"]

        with self.assertRaises(RuntimeError):
            self._reset_with(results)

    def test_one_short_key_is_enough_to_refuse(self):
        """Every key is counted, not just the first.

        A results dict is only ever as long as its shortest column, and reading
        one key to speak for the rest would miss a partial write.
        """
        results = _results_for(BALLOONS)
        results["lon0"] = results["lon0"][:-1]

        with self.assertRaises(RuntimeError):
            self._reset_with(results)

    def test_a_complete_result_set_still_builds_the_flights(self):
        """The other half, or refusing everything would pass the tests above."""
        env = self._reset_with(_results_for(BALLOONS))

        self.assertEqual(env._balloon_flights.shape, (BALLOONS, 6, TIMESTEPS))
        # Distinct balloons, which is what the short-result case destroys.
        self.assertFalse(
            np.array_equal(env._balloon_flights[0], env._balloon_flights[-1])
        )


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheWorkspaceInspectionCannotThrow(unittest.TestCase):
    """It runs inside an ``except`` block, so it has to answer rather than raise.

    Python would chain an error raised here onto the failure being handled, so
    the original stays readable, but the workspace path would never be logged
    and an empty directory would never be removed.
    """

    def test_an_empty_directory_has_nothing_to_keep(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(_monte_carlo_workspace_has_files(directory))

    def test_a_directory_with_a_file_has_something_to_keep(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "balloon_sim.errors.txt").write_text("boom")
            self.assertTrue(_monte_carlo_workspace_has_files(directory))

    def test_an_unreadable_directory_answers_keep_rather_than_raising(self):
        with patch("os.listdir", side_effect=OSError("gone")):
            self.assertTrue(_monte_carlo_workspace_has_files("/nonexistent"))


if __name__ == "__main__":
    unittest.main()
