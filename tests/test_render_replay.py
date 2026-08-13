"""The replay page must render a submission and must not leak one.

A submission carries ``team.secret`` and the agent's source, and a replay page
exists to be shared, so the test that matters most here is the one asserting
neither reaches the output.

Nothing in this file needs the simulation stack: the script reads a submission
as JSON and writes HTML.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_replay import main, render  # noqa: E402

SECRET = "3b4b84252bc53eb1f4d8ea008a9243040088e71a1f1fd7a9ccfe203f9c9cb164"
AGENT_SOURCE = "class MySecretAgent:  # the whole strategy lives here\n    pass\n"


def submission(steps=4, balloons=3, unlaunched=0):
    """A submission shaped like the packer's output.

    ``unlaunched`` leading steps carry ``None`` for every rocket state, which is
    what ``_json_safe`` writes before launch and what every real run begins
    with.
    """
    trajectories = []
    for step in range(steps):
        launched = step >= unlaunched
        trajectories.append(
            {
                "time": round(step * 0.01, 2),
                "rocket_states": [
                    float(step),
                    0.0,
                    20.0 + step,
                    3.0,
                    4.0,
                    12.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                if launched
                else [None] * 13,
                "balloon_states": [
                    [float(i), float(i), 20.0 + i, 0.0, 0.0, 1.0]
                    for i in range(balloons)
                ],
                "balloon_status": [step % 3 for _ in range(balloons)],
            }
        )
    return {
        "format_version": 1,
        "team": {"name": "example", "secret": SECRET},
        "leaderboard_info": {
            "team_name": "example",
            "timestamp_utc": "20260813T000000Z",
            "agent_name": "MySecretAgent",
            "scenario_number": 1,
            "final_reward": 2,
            "random_seed": 7,
        },
        "balloon_world_data": {
            "scenario_parameters": {"environment": {"elevation": 20.0}},
            "trajectories": trajectories,
            "balloon_release_at_step": [0, 1, 2],
        },
        "agent_info": {
            "eval_cfg": {"team_secret": SECRET},
            "agent_module_file": AGENT_SOURCE,
        },
    }


class TheReplayPageKeepsTheSubmissionsSecrets(unittest.TestCase):
    def test_the_team_secret_is_not_in_the_page(self):
        self.assertNotIn(SECRET, render(submission(), 1))

    def test_the_agent_source_is_not_in_the_page(self):
        page = render(submission(), 1)
        self.assertNotIn("the whole strategy lives here", page)
        self.assertNotIn("MySecretAgent:", page)

    def test_a_secret_that_reached_the_page_would_stop_the_write(self):
        """The read-back is the guard, so it has to fail when it should.

        Proved by making the secret something the page necessarily contains,
        rather than by weakening the builder: if the check were removed or
        reordered this test goes green and the real one may not.
        """
        leaky = submission()
        leaky["team"]["secret"] = "example"  # also the team name, which is rendered
        with self.assertRaises(AssertionError):
            render(leaky, 1)


class TheReplayPageCarriesTheRun(unittest.TestCase):
    def test_every_step_is_a_frame_at_stride_one(self):
        page = render(submission(steps=4), 1)
        self.assertIn('"frames":[', page.replace(" ", ""))
        self.assertEqual(page.count('"t":'), 4)

    def test_stride_samples_rather_than_truncates(self):
        page = render(submission(steps=10), 5)
        self.assertEqual(page.count('"t":'), 2)

    def test_speed_is_carried_rather_than_left_to_the_browser(self):
        # 3, 4, 12 -> 13.0, and a stride would make a differenced speed wrong.
        self.assertIn('"v":13.0', render(submission(steps=2), 1))

    def test_the_ground_comes_from_the_scenario_elevation(self):
        self.assertIn('"ground":20.0', render(submission(), 1))

    def test_a_run_with_no_trajectory_says_so(self):
        empty = submission()
        empty["balloon_world_data"]["trajectories"] = []
        with self.assertRaises(ValueError):
            render(empty, 1)


class TheStepsBeforeLaunch(unittest.TestCase):
    """Every real run starts with them, so they are the common case, not an edge.

    ``_json_safe`` turns the pre-launch ``NaN`` rocket states into ``null``. A
    viewer that assumes numbers there raised ``TypeError`` on the first frame of
    the first real submission this was pointed at.
    """

    def test_a_pre_launch_frame_is_kept_and_carries_no_rocket(self):
        page = render(submission(steps=4, unlaunched=2), 1)
        self.assertEqual(page.count('"r":null'), 2)
        self.assertEqual(page.count('"t":'), 4)

    def test_a_pre_launch_frame_still_carries_its_balloons(self):
        page = render(submission(steps=2, balloons=3, unlaunched=2), 1)
        self.assertEqual(page.count('"r":null'), 2)
        self.assertIn('"s":[', page)

    def test_the_pad_comes_from_the_first_step_that_has_a_rocket(self):
        # Step 2 is the first launched one, and its x is float(step).
        self.assertIn('"pad":[2.0,0.0]', render(submission(steps=4, unlaunched=2), 1))

    def test_a_run_that_never_launched_still_renders(self):
        page = render(submission(steps=3, unlaunched=3), 1)
        self.assertIn('"pad":[0.0,0.0]', page)
        self.assertEqual(page.count('"r":null'), 3)


class TheCommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_render_replay_tmp"
        self.tmp.mkdir(exist_ok=True)
        self.path = self.tmp / "submission.json"
        self.path.write_text(json.dumps(submission()), encoding="utf-8")

    def tearDown(self):
        for path in self.tmp.iterdir():
            path.unlink()
        self.tmp.rmdir()

    def test_the_output_lands_next_to_the_input_by_default(self):
        self.assertEqual(main([str(self.path)]), 0)
        self.assertTrue((self.tmp / "submission.html").exists())

    def test_an_unsupported_format_version_is_refused(self):
        payload = submission()
        payload["format_version"] = 2
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit):
            main([str(self.path)])

    def test_a_stride_below_one_is_refused(self):
        with self.assertRaises(SystemExit):
            main([str(self.path), "--stride", "0"])


if __name__ == "__main__":
    unittest.main()
