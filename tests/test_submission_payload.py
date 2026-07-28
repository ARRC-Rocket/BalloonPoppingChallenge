"""The submission no longer ships ``balloon_flights`` (issue #57).

That array was 71% of a scenario-1 submission and nothing read it. Dropping it
rests on one claim, so this pins the claim rather than only the removal: the
balloon data the array holds is already in ``trajectories``, one step across.

Only ``rocketpy`` is guarded: a missing simulation stack is a legitimate skip,
but a broken import inside this package is a failure and must stay loud.
"""

import unittest
from importlib.util import find_spec
from unittest import mock

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.evaluation.results import utils

SCENARIO_NUMBER = 0


class _FakeEnv:
    """The attributes ``pack_for_submission`` reads, and nothing else."""

    _popped_count = 3
    trajectories = [{"balloon_states": [[0.0] * 6]}]
    _balloon_release_at_step = [0]
    _rocket_flight = None
    _balloon_flights = np.zeros((1, 6, 2))


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestBalloonFlightsIsNotShipped(unittest.TestCase):
    def _payload(self):
        captured = {}
        eval_cfg = {
            "team_name": "unittest_team",
            "team_secret": "secret",
            "agent_module_path": __file__,
            "agent_name": "agent",
            "scenario_number": SCENARIO_NUMBER,
        }

        with (
            mock.patch.object(
                utils.urllib.request, "urlopen", side_effect=OSError("offline")
            ),
            mock.patch.object(
                utils.pickle, "dump", lambda payload, _file: captured.update(payload)
            ),
            mock.patch.object(
                utils, "open", mock.mock_open(read_data=b""), create=True
            ),
        ):
            utils.pack_for_submission(eval_cfg, _FakeEnv(), {"scenario": {}})

        return captured

    def test_the_array_is_absent(self):
        world = self._payload()["balloon_world_data"]

        self.assertNotIn("balloon_flights", world)

    def test_the_other_balloon_world_fields_are_unchanged(self):
        """Scoped to "this change removed one key", not to a consumer contract.

        Of these four, ``trajectories`` is the one the leaderboard actually
        reads. Naming the test after the consumer would have frozen the other
        three as requirements and made the next payload trim argue with it.
        """
        world = self._payload()["balloon_world_data"]

        for key in [
            "scenario_parameters",
            "trajectories",
            "balloon_release_at_step",
            "rocket_flight",
        ]:
            with self.subTest(key=key):
                self.assertIn(key, world)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTrajectoriesStillCarryTheBalloonData(unittest.TestCase):
    """``trajectories[k]["balloon_states"]`` is ``_balloon_flights[:, :, k + 1]``.

    This is the reason the array can go, so it is worth a test of its own. If
    ``step`` ever stops recording what it indexed, or the offset moves, this
    fails and says that dropping the array is no longer free.

    Scenario 0's balloons are static, so its real flight array is identical at
    every timestep and would hide any offset at all. A ramp is written over it
    first so each timestep is distinguishable, which is what gives this teeth.
    """

    def test_each_record_matches_the_flight_array_one_step_across(self):
        scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        env.reset(seed=scenario_params["scenario"]["random_seed"])

        flights = np.zeros_like(np.asarray(env._balloon_flights, dtype=float))
        flights += np.arange(flights.shape[2], dtype=float)
        env._balloon_flights = flights

        action = env.action_space.sample()
        action["launch"] = np.array(0, dtype=action["launch"].dtype)
        for key in ("tvc", "throttle", "roll"):
            action[key] = np.zeros_like(action[key])

        for _ in range(5):
            env.step(action)

        self.assertEqual(len(env.trajectories), 5)
        for step_index, record in enumerate(env.trajectories):
            with self.subTest(step=step_index):
                np.testing.assert_array_equal(
                    np.asarray(record["balloon_states"], dtype=float),
                    flights[:, :, step_index + 1],
                )

    def test_the_ramp_would_expose_a_wrong_offset(self):
        # Guards the guard: if the ramp were flat the assertion above would hold
        # for any offset, which is exactly how scenario 0's real array behaves.
        flights = np.zeros((2, 6, 4)) + np.arange(4, dtype=float)

        self.assertFalse(np.array_equal(flights[:, :, 1], flights[:, :, 2]))


if __name__ == "__main__":
    unittest.main()
