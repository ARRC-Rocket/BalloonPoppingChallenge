"""The submission no longer ships ``balloon_flights`` (issue #57).

That array was 71% of a scenario-1 submission and nothing read it. Dropping it
rests on one claim, so this pins the claim rather than only the removal: the
balloon data the array holds is already in ``trajectories``, one step across.

What a submission contains is asked of ``build_submission_payload`` directly.
The version this replaced captured it by mocking ``pickle.dump``, which tied the
contents to the encoding and would have broken on #58 replacing the encoder with
JSON. It also mocked ``utils.open``, which is not the write path:
``pack_for_submission`` goes through ``_write_atomically`` to ``mkstemp``,
``fsync`` and ``os.replace``, so with ``dump`` doing nothing the write still ran
and renamed an empty file into place. Every call left a finished-looking zero
byte submission in the package directory, 130 across the worktrees on one
machine. That the writer is reached at all is now its own test, with the writer
mocked at its own boundary.

Only ``rocketpy`` is guarded: a missing simulation stack is a legitimate skip,
but a broken import inside this package is a failure and must stay loud.
"""

import copy
import unittest
from datetime import datetime, timezone
from importlib.util import find_spec
from unittest import mock

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.evaluation.results import utils

SCENARIO_NUMBER = 0
STEPS_TO_CHECK = 5

# What the world section holds, exactly. An extra key is as much a change to the
# format as a missing one, and #57 was about not shipping the array rather than
# about not shipping that spelling of its name.
EXPECTED_WORLD_KEYS = frozenset(
    {
        "scenario_parameters",
        "trajectories",
        "balloon_release_at_step",
        "rocket_flight",
    }
)

EVAL_CFG = {
    "team_name": "unittest_team",
    "team_secret": "secret",
    "agent_name": "agent",
    "scenario_number": SCENARIO_NUMBER,
}


class _FakeEnv:
    """The attributes ``pack_for_submission`` reads, and nothing else."""

    _popped_count = 3
    np_random_seed = 0
    trajectories = [{"balloon_states": [[0.0] * 6]}]
    _balloon_release_at_step = [0]
    _rocket_flight = None
    _balloon_flights = np.zeros((1, 6, 2))


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestBalloonFlightsIsNotShipped(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = utils.build_submission_payload(
            dict(EVAL_CFG, agent_module_path=__file__),
            _FakeEnv(),
            {"scenario": {}},
            datetime(2026, 7, 30, tzinfo=timezone.utc),
        )

    def test_the_array_is_absent(self):
        self.assertNotIn("balloon_flights", self.payload["balloon_world_data"])

    def test_the_world_section_holds_exactly_the_remaining_fields(self):
        """An exact set, not four membership checks.

        Membership alone accepted a renamed copy of the same array, which is the
        thing #57 was about, and accepted any of the four being replaced by
        ``None``.
        """
        self.assertEqual(set(self.payload["balloon_world_data"]), EXPECTED_WORLD_KEYS)

    def test_the_field_the_leaderboard_reads_is_the_real_one(self):
        """Of the four, ``trajectories`` is what is actually consumed."""
        self.assertIs(
            self.payload["balloon_world_data"]["trajectories"], _FakeEnv.trajectories
        )

    def test_the_agent_source_is_text(self):
        """Production reads the file as ``str``. The mock this replaced supplied
        bytes, which pickle accepts and JSON does not, so #58 would have failed
        on the fixture rather than on anything real."""
        self.assertIsInstance(self.payload["agent_info"]["agent_module_file"], str)

    def test_packing_writes_through_the_atomic_writer_and_nowhere_else(self):
        """The failure that prompted the rewrite, kept as a test.

        Asserted on the writer rather than on the directory, because results/ is
        in .gitignore: the leftover files never appeared in ``git status``, so a
        clean-worktree check in CI would not have caught this either.
        """
        with (
            mock.patch.object(utils, "_write_atomically") as writer,
            mock.patch.object(utils, "_check_evaluate_integrity"),
        ):
            utils.pack_for_submission(
                dict(EVAL_CFG, agent_module_path=__file__),
                _FakeEnv(),
                {"scenario": {}},
            )

        self.assertEqual(writer.call_count, 1)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTrajectoriesStillCarryTheBalloonData(unittest.TestCase):
    """``trajectories[k]["balloon_states"]`` is ``_balloon_flights[:, :, k + 1]``.

    This is the reason the array can go, so it is worth a test of its own. If
    ``step`` ever stops recording what it indexed, or the offset moves, this
    fails and says that dropping the array is no longer free.

    Scenario 0's balloons are static, so its real flight array is identical at
    every timestep and would hide any offset. A pattern is written over it that
    is distinct along all three axes, so a swapped balloon, a swapped state
    component and a shifted timestep each fail separately. The ramp this
    replaced varied with time alone and passed with the balloon order reversed.
    """

    def test_each_record_matches_the_flight_array_one_step_across(self):
        scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
        env = BalloonPoppingEnv(
            render_mode=None, parameters=copy.deepcopy(scenario_params)
        )
        self.addCleanup(env.close)
        env.reset(seed=scenario_params["scenario"]["random_seed"])

        shape = np.asarray(env._balloon_flights).shape
        self.assertGreaterEqual(
            shape[2],
            STEPS_TO_CHECK + 1,
            "the scenario is too short for the offset this checks",
        )
        balloon, component, timestep = np.indices(shape, dtype=float)
        flights = balloon * 1e6 + component * 1e4 + timestep

        # The fixture's own teeth, on the array the assertion below uses rather
        # than on a separate toy one. Dropping the pattern used to leave a guard
        # test green while every offset passed.
        self.assertFalse(np.array_equal(flights[:, :, 1], flights[:, :, 2]))
        self.assertFalse(np.array_equal(flights[0], flights[-1]))
        self.assertFalse(np.array_equal(flights[:, 0, :], flights[:, 1, :]))

        env._balloon_flights = flights

        action = env.action_space.sample()
        for key in action:
            action[key] = np.zeros_like(action[key])

        for _ in range(STEPS_TO_CHECK):
            env.step(action)

        self.assertEqual(len(env.trajectories), STEPS_TO_CHECK)
        for step_index, record in enumerate(env.trajectories):
            with self.subTest(step=step_index):
                np.testing.assert_array_equal(
                    np.asarray(record["balloon_states"], dtype=float),
                    flights[:, :, step_index + 1],
                )


if __name__ == "__main__":
    unittest.main()
