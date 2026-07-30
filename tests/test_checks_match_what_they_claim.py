"""Three places where the tool claimed more than it did.

The writer promised strict JSON and left the encoder's default in place, which
writes bare ``NaN``. The checker's comment said an invalid rocket path is
refused before the hundred-flight rebuild, and the rebuild ran anyway. And the
producer stamps ``format_version``, which the checker never read.

Only ``rocketpy`` is guarded: a missing simulation stack is a legitimate skip,
but a broken import inside this package is a failure and must stay loud.
"""

import json
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import verify_submission as verifier
    from BalloonPoppingGymEnv.evaluation.results import utils


def _submission(**overrides):
    submission = {
        "format_version": 1,
        "leaderboard_info": {"scenario_number": 0},
        "balloon_world_data": {
            "scenario_parameters": {"scenario": {"number": 0, "random_seed": 0}},
            "trajectories": [{"rocket_states": [0.0] * 13}],
            "balloon_release_at_step": [0],
        },
    }
    submission.update(overrides)
    return submission


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheFormatVersionIsChecked(unittest.TestCase):
    """A file with no version, or version 0 renamed to `.json`, used to reach
    every check below as though it were the format they describe."""

    def _refused_by_the_version_check(self, submission):
        """The named finding, and nothing after it.

        Asserting only that something failed proves nothing: this fixture is a
        stub scenario, so the parameter comparison fails on its own and the test
        passes with the version check deleted. It is the finding and the early
        return that have to be pinned.
        """
        with mock.patch.object(verifier, "check_scenario_is_official") as official:
            findings = verifier.verify(submission, verifier.DEFAULT_TOLERANCE_METRES)
        failed = [f for f in findings if not f.ok]
        return (
            len(failed) == 1
            and failed[0].name == "submission format version"
            and not official.called
        )

    def test_a_missing_version_is_refused(self):
        submission = _submission()
        del submission["format_version"]

        self.assertTrue(self._refused_by_the_version_check(submission))

    def test_the_previous_format_is_refused(self):
        """Version 0 is the pickle payload. Renaming it to `.json` does not
        make it one."""
        self.assertTrue(
            self._refused_by_the_version_check(_submission(format_version=0))
        )

    def test_a_future_version_is_refused(self):
        self.assertTrue(
            self._refused_by_the_version_check(_submission(format_version=2))
        )

    def test_a_boolean_is_not_a_version(self):
        """`True == 1`, so a plain equality test would have accepted it."""
        self.assertTrue(
            self._refused_by_the_version_check(_submission(format_version=True))
        )

    def test_the_version_it_writes_gets_past_this_check(self):
        """The control. Refusing every version would pass the four above."""
        with mock.patch.object(verifier, "check_scenario_is_official") as official:
            official.return_value = ([], None)
            with mock.patch.object(
                verifier, "check_internal_consistency", return_value=[]
            ):
                findings = verifier.verify(
                    _submission(), verifier.DEFAULT_TOLERANCE_METRES
                )

        self.assertTrue(official.called)
        self.assertTrue(all(f.ok for f in findings), [str(f) for f in findings])

    def test_the_version_it_writes_is_the_version_it_accepts(self):
        """These two drifting apart is the whole failure this guards against."""
        self.assertEqual(verifier._SUPPORTED_FORMAT_VERSION, 1)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestAnInvalidPathIsRefusedBeforeTheRebuild(unittest.TestCase):
    def test_the_monte_carlo_does_not_run_for_a_path_that_is_not_one(self):
        """Scenario 1 rebuilds a hundred flights. A file that cannot survive the
        cheap check should not be paying for that."""
        submission = _submission()
        submission["leaderboard_info"]["scenario_number"] = 1
        submission["balloon_world_data"]["scenario_parameters"]["scenario"][
            "number"
        ] = 1

        with (
            mock.patch.object(verifier, "check_scenario_is_official") as official,
            mock.patch.object(
                verifier, "check_the_records_are_the_right_shape", return_value=[]
            ),
            mock.patch.object(
                verifier,
                "check_the_rocket_path_is_a_trajectory",
                return_value=[
                    verifier.Finding("rocket path", False, "not a trajectory")
                ],
            ),
            mock.patch.object(verifier, "_regenerate_balloon_flights") as rebuild,
        ):
            official.return_value = ([], {"scenario": {"random_seed": 0}})
            findings = verifier.verify(submission, verifier.DEFAULT_TOLERANCE_METRES)

        rebuild.assert_not_called()
        self.assertTrue(any(not f.ok for f in findings))

    def test_a_good_path_still_reaches_the_rebuild(self):
        """The control. Returning early always would pass the test above."""
        submission = _submission()

        with (
            mock.patch.object(verifier, "check_scenario_is_official") as official,
            mock.patch.object(
                verifier, "check_the_records_are_the_right_shape", return_value=[]
            ),
            mock.patch.object(
                verifier, "check_the_rocket_path_is_a_trajectory", return_value=[]
            ),
            mock.patch.object(verifier, "_regenerate_balloon_flights") as rebuild,
            mock.patch.object(verifier, "_release_eligibility", return_value=None),
            mock.patch.object(verifier, "check_internal_consistency", return_value=[]),
            mock.patch.object(verifier, "check_balloon_trajectories", return_value=[]),
        ):
            rebuild.return_value = (None, None, None)
            official.return_value = ([], {"scenario": {"random_seed": 0}})
            verifier.verify(submission, verifier.DEFAULT_TOLERANCE_METRES)

        rebuild.assert_called_once()


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheWriterRefusesNonFiniteValues(unittest.TestCase):
    """`_json_safe` covers the payload as it is today. The encoder is the
    boundary that has to hold when a field is added that it does not reach."""

    def _pack(self, tmp, sanitize):
        env = SimpleNamespace(
            _popped_count=1,
            np_random_seed=0,
            trajectories=[{"rocket_states": [float("nan")]}],
            _balloon_release_at_step=[0],
            _rocket_flight=None,
            _balloon_flights=[[[0.0]]],
        )
        agent = tmp / "agent.py"
        agent.write_text("# agent\n", encoding="utf-8")
        eval_cfg = {
            "team_name": "t",
            "team_secret": "s",
            "agent_module_path": str(agent),
            "agent_name": "a",
            "scenario_number": 0,
        }
        with (
            mock.patch.object(utils, "_json_safe", sanitize),
            mock.patch.object(utils, "_check_evaluate_integrity"),
            mock.patch.object(utils.os.path, "dirname", return_value=str(tmp)),
        ):
            utils.pack_for_submission(eval_cfg, env, {"scenario": {"number": 0}})

    def test_a_payload_the_sanitizer_missed_fails_instead_of_being_written(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            with self.assertRaises(ValueError):
                self._pack(tmp, lambda payload: payload)
            self.assertEqual(list(tmp.glob("*_submission.json")), [])

    def test_the_sanitized_payload_still_writes(self):
        """The control. Refusing everything would pass the test above."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self._pack(tmp, utils._json_safe)
            written = list(tmp.glob("*_submission.json"))
            self.assertEqual(len(written), 1)
            json.loads(written[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
