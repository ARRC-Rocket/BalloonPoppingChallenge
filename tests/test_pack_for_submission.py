"""Tests for ``pack_for_submission`` writing a safe JSON submission.

The submission is uploaded to an unauthenticated endpoint, so it must be plain
JSON with no material that a later ``dill.loads`` could execute. These drive the
real ``pack_for_submission`` (the network checksum call is mocked, the file it
writes is read back), so the new ``json.dump`` path is actually exercised.
"""

import glob
import json
import os
import re
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import mock

_HAS_ROCKETPY = find_spec("rocketpy") is not None

_HEX_BLOB = re.compile(r"[0-9a-f]{200,}")


@unittest.skipUnless(_HAS_ROCKETPY, "requires the rocketpy simulation stack")
class TestPackForSubmission(unittest.TestCase):
    def setUp(self):
        from rocketpy import Function

        from BalloonPoppingGymEnv.evaluation.results import utils

        self.utils = utils
        self.results_dir = os.path.dirname(utils.__file__)

        # A Function with a callable source is what the encoder would hex-encode
        # with dill under the default allow_pickle=True.
        self.env = SimpleNamespace(
            _popped_count=5,
            trajectories=[
                {
                    "rocket_states": [1.0, float("nan")],
                    "balloon_states": [[0.1] * 6],
                    "balloon_status": [1],
                }
            ],
            _balloon_release_at_step=[0, 1],
            _rocket_flight=Function(lambda t: t * 2.0),
            _balloon_flights=[[[0.0] * 3]],
        )

        self.agent_file = os.path.join(self.results_dir, "_unittest_agent_module.py")
        with open(self.agent_file, "w", encoding="utf-8") as handle:
            handle.write("# unit-test agent source\n")

        self.eval_cfg = {
            "team_name": "unittest_team",
            "team_secret": "s3cr3t",
            "agent_name": "A",
            "scenario_number": 0,
            "agent_module_path": self.agent_file,
        }
        self.created = []

    def tearDown(self):
        for path in self.created:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.agent_file):
            os.remove(self.agent_file)

    def _pack(self):
        """Run the real ``pack_for_submission`` and return the file it wrote."""
        local = os.path.join(os.path.dirname(self.results_dir), "evaluate.py")
        with open(local, "rb") as handle:
            local_bytes = handle.read()

        before = set(glob.glob(os.path.join(self.results_dir, "*_submission.json")))
        with mock.patch.object(self.utils.urllib.request, "urlopen") as urlopen:
            # match the local checksum so the integrity check stays quiet
            urlopen.return_value.__enter__.return_value.read.return_value = local_bytes
            self.utils.pack_for_submission(self.eval_cfg, self.env, {"scenario": {}})
        after = set(glob.glob(os.path.join(self.results_dir, "*_submission.json")))

        new = after - before
        self.assertEqual(len(new), 1, "expected exactly one submission file")
        path = new.pop()
        self.created.append(path)
        return path

    def _load(self):
        with open(self._pack(), encoding="utf-8") as handle:
            return json.load(handle)

    def test_writes_json_extension(self):
        self.assertTrue(self._pack().endswith(".json"))

    def test_required_fields_and_format_version(self):
        data = self._load()
        self.assertEqual(data["format_version"], 1)
        self.assertEqual(data["team"]["name"], "unittest_team")
        self.assertEqual(data["leaderboard_info"]["team_name"], "unittest_team")
        self.assertEqual(data["leaderboard_info"]["final_reward"], 5)
        self.assertIn("trajectories", data["balloon_world_data"])
        self.assertIn("eval_cfg", data["agent_info"])

    def test_rocket_flight_is_not_double_encoded(self):
        # inlined as a JSON object, not a json.dumps-ed string
        self.assertNotIsInstance(
            self._load()["balloon_world_data"]["rocket_flight"], str
        )

    def test_no_dill_executable_material(self):
        from rocketpy._encoders import RocketPyEncoder

        # teeth: this Function does hex-encode with dill under the default
        default = json.dumps(self.env._rocket_flight, cls=RocketPyEncoder)
        self.assertRegex(default, _HEX_BLOB, "test Function should trigger a dill blob")

        # the producer passes allow_pickle=False, so its file must carry none
        with open(self._pack(), encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotRegex(raw, _HEX_BLOB)


if __name__ == "__main__":
    unittest.main()
