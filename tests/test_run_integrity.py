"""A run that never finished must not pack as a finished one.

``build_submission_payload`` reads the score and the trajectories straight off
the environment. Nothing said whether the episode had reached an ending, so a
run stopped part way through produced a submission indistinguishable from a
complete one, carrying whatever score it had reached by then.

Measured before the fix: five steps into scenario 0, with the rocket still on
the pad, the payload came out fully formed and ``leaderboard_info`` had no field
that mentioned the run being unfinished.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import unittest
from datetime import datetime, timezone
from importlib.util import find_spec

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.evaluation.results.utils import (
        build_submission_payload,
        pack_for_submission,
    )

SCENARIO_NUMBER = 0

_EVAL_CFG = {
    "team_name": "integrity test",
    "team_secret": "not a real secret",
    "agent_name": "idle",
    "scenario_number": SCENARIO_NUMBER,
    "agent_module_path": "BalloonPoppingGymEnv/agents/example_agents.py",
}


def _idle_action():
    """Never launch, so the episode runs to its timeout rather than a flight."""
    return {
        "launch": False,
        "launch_inclination_heading": [0.0, 0.0],
        "tvc": [0.0, 0.0],
        "throttle": 0.0,
        "roll": 0.0,
    }


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class RunIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        self.env = BalloonPoppingEnv(render_mode=None, parameters=self.parameters)
        self.env.reset(seed=self.parameters["scenario"]["random_seed"])

    def _payload(self):
        return build_submission_payload(
            _EVAL_CFG, self.env, self.parameters, datetime.now(timezone.utc)
        )

    def _run_to_the_end(self):
        while self.env._episode_ending is None:
            self.env.step(_idle_action())

    def test_a_run_that_stopped_part_way_is_marked_unfinished(self):
        for _ in range(5):
            self.env.step(_idle_action())

        info = self._payload()["leaderboard_info"]

        self.assertIsNone(info["episode_ending"])
        self.assertEqual(info["steps_run"], 5)

    def test_packing_a_run_that_never_ended_is_refused(self):
        for _ in range(5):
            self.env.step(_idle_action())

        with self.assertRaises(RuntimeError) as raised:
            pack_for_submission(_EVAL_CFG, self.env, self.parameters)

        self.assertIn("never reached an ending", str(raised.exception))
        self.assertIn("5 steps", str(raised.exception))

    def test_a_finished_run_says_how_it_ended(self):
        """The control. Without it the two above pass on a check that refuses
        every run, which would take the competition down rather than protect it.
        """
        self._run_to_the_end()

        info = self._payload()["leaderboard_info"]

        # Which ending scenario 0 reaches is the scoring question in #104 and is
        # not decided here. What this pins is that an ending was recorded, and
        # that the step count is the whole episode rather than a prefix of it.
        self.assertIn(info["episode_ending"], ("terminated", "truncated"))
        self.assertGreater(info["steps_run"], 5)
        self.assertEqual(info["steps_run"], self.env.current_step)

    def test_the_ending_does_not_survive_a_reset(self):
        """A second episode on the same environment starts unfinished again, or
        one completed run would let every later partial run pack."""
        self._run_to_the_end()
        self.assertIsNotNone(self.env._episode_ending)

        self.env.reset(seed=self.parameters["scenario"]["random_seed"])

        self.assertIsNone(self.env._episode_ending)
        with self.assertRaises(RuntimeError):
            pack_for_submission(_EVAL_CFG, self.env, self.parameters)


if __name__ == "__main__":
    unittest.main()
