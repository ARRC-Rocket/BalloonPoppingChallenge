"""Tests for ``pack_for_submission`` writing a safe JSON submission.

The submission is uploaded to an unauthenticated endpoint, so it must be plain
JSON with no material that a later ``dill.loads`` could execute. These drive the
real ``pack_for_submission`` (the network checksum call is mocked, the file it
writes is read back), so the new ``json.dump`` path is actually exercised.
"""

import glob
import json
import math
import os
import re
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import mock

import numpy as np

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
            _episode_ending="terminated",
            current_step=123,
            np_random_seed=0,
            trajectories=[
                {
                    "rocket_states": [1.0, float("nan")],
                    "balloon_states": [[0.1] * 6],
                    "balloon_status": [1],
                }
            ],
            # A float64 array with one non-finite entry, so the substituting
            # branch is exercised as well as the plain float one. On the
            # release schedule rather than on `_balloon_flights`, which the
            # payload has not carried since #57.
            _balloon_release_at_step=np.array([0.0, np.nan, 2.0]),
            _rocket_flight=Function(lambda t: t * 2.0),
            # A real balloon flight is a float64 array. One non-finite entry sends
            # it down the substituting branch rather than the untouched one.
            _balloon_flights=np.array([[[0.0, np.nan, 2.0]]]),
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
        # Snapshot rather than a list built as files are recognised. The list
        # only grew on the success path, so a failing assertion in `_pack` left
        # the file it had just written in the package directory; a mutation run
        # left eight of them there.
        self.before = self._submissions()

    def _submissions(self):
        return set(glob.glob(os.path.join(self.results_dir, "*_submission.*")))

    def tearDown(self):
        for path in self._submissions() - self.before:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.agent_file):
            os.remove(self.agent_file)

    def _pack(self):
        """Run the real ``pack_for_submission`` and return the file it wrote."""
        local = os.path.join(os.path.dirname(self.results_dir), "evaluate.py")
        with open(local, "rb") as handle:
            local_bytes = handle.read()

        before = self._submissions()
        with mock.patch.object(self.utils.urllib.request, "urlopen") as urlopen:
            # match the local checksum so the integrity check stays quiet
            urlopen.return_value.__enter__.return_value.read.return_value = local_bytes
            self.utils.pack_for_submission(self.eval_cfg, self.env, {"scenario": {}})

        # Any extension, so a run that writes the wrong one fails on the name
        # rather than on having written nothing, and tearDown still removes it.
        new = self._submissions() - before
        self.assertEqual(len(new), 1, "expected exactly one submission file")
        return new.pop()

    def _load(self):
        with open(self._pack(), encoding="utf-8") as handle:
            return json.load(handle)

    def test_the_flight_array_is_still_not_shipped(self):
        """The fixture holds one, and #57 is about it not reaching the file.

        A format change is exactly when a dropped field comes back by accident,
        and every other assertion here would pass if it had.
        """
        self.assertNotIn("balloon_flights", self._load()["balloon_world_data"])

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

    def test_the_file_is_strict_json(self):
        """No ``NaN`` or ``Infinity`` tokens, which RFC 8259 has no room for.

        Python reads them back either way, so the leaderboard would not notice,
        but a file named ``.json`` should be one that ``jq`` or a browser can
        open. ``parse_constant`` fires on exactly those three tokens and on
        nothing else, so it answers the question directly rather than by
        searching the text for a substring that could appear inside a string.
        """
        with open(self._pack(), encoding="utf-8") as handle:
            raw = handle.read()

        found = []
        json.loads(raw, parse_constant=found.append)
        self.assertEqual(found, [], "submission is not valid JSON")

    def test_a_non_finite_value_inside_the_flight_is_substituted(self):
        """The flight is encoded to text, so the walk cannot reach into it.

        ``pack_for_submission`` round-trips it back to plain objects first. A
        Function built from a source array keeps that array in its encoded form,
        which is where a real flight's own non-finite values sit.
        """
        from rocketpy import Function

        self.env._rocket_flight = Function([[0.0, float("nan")], [1.0, 2.0]])

        with open(self._pack(), encoding="utf-8") as handle:
            raw = handle.read()

        found = []
        json.loads(raw, parse_constant=found.append)
        self.assertEqual(found, [], "a value inside the flight reached the file")

    def test_non_finite_values_become_null(self):
        data = self._load()

        # a plain Python float in the trajectory records
        self.assertEqual(
            data["balloon_world_data"]["trajectories"][0]["rocket_states"], [1.0, None]
        )
        # and a float64 array, which takes the other branch
        self.assertEqual(
            data["balloon_world_data"]["balloon_release_at_step"],
            [0.0, None, 2.0],
        )


@unittest.skipUnless(_HAS_ROCKETPY, "requires the rocketpy simulation stack")
class TestJsonSafe(unittest.TestCase):
    """The substitution itself, including what it must leave alone."""

    def setUp(self):
        from BalloonPoppingGymEnv.evaluation.results import utils

        self._json_safe = utils._json_safe

    def test_finite_values_are_untouched(self):
        for value in [0.0, -1.5, 1e300, 7, "NaN", None, True, [1.0, 2.0], {"a": 1.0}]:
            with self.subTest(value=value):
                self.assertEqual(self._json_safe(value), value)

    def test_non_finite_scalars_become_none(self):
        for value in [float("nan"), float("inf"), float("-inf"), np.float64("nan")]:
            with self.subTest(value=repr(value)):
                self.assertIsNone(self._json_safe(value))

    def test_a_clean_float_array_is_left_as_an_array(self):
        # Converting it would trade tens of megabytes of float64 for individually
        # allocated Python floats, and the encoder can stream the array as it is.
        array = np.array([1.0, 2.0, 3.0])
        self.assertIs(self._json_safe(array), array)

    def test_an_array_holding_non_finite_values_is_substituted(self):
        result = self._json_safe(np.array([[1.0, np.nan], [np.inf, -np.inf]]))
        self.assertEqual(result, [[1.0, None], [None, None]])

    def test_nesting_is_followed(self):
        result = self._json_safe(
            {"a": [{"b": (float("nan"), 1.0)}], "c": np.array([np.nan])}
        )
        self.assertEqual(result, {"a": [{"b": [None, 1.0]}], "c": [None]})

    def test_a_non_numeric_array_is_handed_on_untouched(self):
        # np.isfinite raises TypeError on anything that is not numeric, so the
        # dtype check is what stops a string array from crashing the pack. The
        # encoder converts it afterwards.
        array = np.array(["a", "b"])
        self.assertIs(self._json_safe(array), array)

    def test_integer_arrays_keep_their_type(self):
        # balloon_status is integer data; there is nothing to substitute and it
        # must not be turned into floats.
        result = self._json_safe(np.array([0, 1, 2]))
        self.assertEqual(list(result), [0, 1, 2])
        self.assertFalse(any(isinstance(value, float) for value in result))

    def test_the_result_survives_a_strict_dump(self):
        payload = {"x": [float("nan"), 1.0], "y": np.array([np.inf, 2.0])}
        # allow_nan=False is the strict encoder: it raises on any non-finite left
        # behind, so this fails loudly if the walk misses one.
        json.dumps(self._json_safe(payload), allow_nan=False)
        self.assertTrue(math.isnan(payload["x"][0]), "the input must not be mutated")


if __name__ == "__main__":
    unittest.main()
