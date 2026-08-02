import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario_1_one_balloon_agent import OneBalloonAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class TestOneBalloonAgent(unittest.TestCase):
    def setUp(self):
        _, self.given = load_scenario_parameters(1)
        self.agent = OneBalloonAgent(self.given)
        balloon_count = self.given["balloon"]["num"]
        self.observation = {
            "simulation_time": 0.0,
            "balloon_status": np.zeros((balloon_count, 1), dtype=int),
            "balloon_states": np.zeros((balloon_count, 6)),
            "rocket_sensors": np.full(12, np.nan),
        }

    def test_waits_with_finite_neutral_action_before_launch(self):
        action = self.agent.get_action(self.observation)

        self.assertFalse(action["launch"])
        np.testing.assert_allclose(action["tvc"], [0.0, 0.0])
        self.assertEqual(action["roll"], 0.0)
        self.assertEqual(action["throttle"], 1.0)

    def test_selects_one_reachable_target_at_launch(self):
        self.observation["simulation_time"] = 49.5
        self.observation["balloon_status"][3, 0] = 1
        self.observation["balloon_states"][3] = [20.0, 10.0, 70.0, 1.0, 0.0, 3.0]

        action = self.agent.get_action(self.observation)

        self.assertTrue(action["launch"])
        self.assertEqual(self.agent.target, 3)
        self.assertTrue(np.all(np.isfinite(action["launch_inclination_heading"])))
        self.assertGreaterEqual(action["launch_inclination_heading"][0], 65.0)


if __name__ == "__main__":
    unittest.main()
