"""A run that drew its own seed has to be reproducible from what it ships.

`random_seed: null` is what the scenario file offers for a random run. The
environment then draws a seed, and the submission recorded the `null` that was
asked for rather than the number that was used. `verify_submission.py` rebuilds
the balloon field by resetting the environment with the seed it finds, so that
lost the only thing it needed.

Through the real `build_submission_payload` rather than a helper, because what
matters is the field a competitor's file actually carries.

Scenario 0 because it skips the Monte Carlo, so the whole file runs in seconds.
The seed and the release schedule are drawn the same way in both scenarios.

Only ``rocketpy`` is guarded: a missing simulation stack is a legitimate skip,
but a broken import inside this package is a failure and must stay loud.
"""

import copy
import datetime
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import verify_submission as verifier
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.evaluation.results.utils import build_submission_payload

SCENARIO_NUMBER = 0
PACKED_AT = datetime.datetime(2026, 8, 1, tzinfo=datetime.timezone.utc)


def _parameters(random_seed):
    loaded = load_scenario_parameters(SCENARIO_NUMBER)
    parameters = copy.deepcopy(loaded[0] if isinstance(loaded, tuple) else loaded)
    parameters["scenario"]["random_seed"] = random_seed
    return parameters


class _Run:
    """A reset environment and the payload a pack would build from it."""

    def __init__(self, test, random_seed):
        self.parameters = _parameters(random_seed)
        self.env = BalloonPoppingEnv(render_mode=None, parameters=self.parameters)
        test.addCleanup(self.env.close)
        self.env.reset(seed=self.parameters["scenario"]["random_seed"])
        directory = Path(tempfile.mkdtemp())
        agent = directory / "agent.py"
        agent.write_text("# agent\n", encoding="utf-8")
        self.payload = build_submission_payload(
            {
                "team_name": "t",
                "team_secret": "s",
                "agent_module_path": str(agent),
                "agent_name": "a",
                "scenario_number": SCENARIO_NUMBER,
            },
            self.env,
            self.parameters,
            PACKED_AT,
        )


def _submission(recorded_seed, configured_seed=None):
    """The two fields the seed check reads, without running anything."""
    return {
        "leaderboard_info": {
            "scenario_number": SCENARIO_NUMBER,
            "random_seed": recorded_seed,
        },
        "balloon_world_data": {"scenario_parameters": _parameters(configured_seed)},
    }


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestThePackerRecordsWhatWasUsed(unittest.TestCase):
    def test_a_null_seed_is_recorded_as_the_number_that_was_drawn(self):
        run = _Run(self, None)

        seed = run.payload["leaderboard_info"]["random_seed"]

        self.assertIsInstance(seed, int)
        self.assertEqual(seed, run.env.np_random_seed)

    def test_an_explicit_seed_is_recorded_as_itself(self):
        """The control. Writing a fresh seed over every run would pass the test
        above and break every submission that asked for a specific world."""
        run = _Run(self, 7)

        self.assertEqual(run.payload["leaderboard_info"]["random_seed"], 7)

    def test_the_parameters_still_carry_what_was_asked_for(self):
        """The two are not the same question, and the checker compares them."""
        run = _Run(self, None)

        scenario = run.payload["balloon_world_data"]["scenario_parameters"]["scenario"]
        self.assertIsNone(scenario["random_seed"])


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheVerifierCanRebuildIt(unittest.TestCase):
    def test_a_recorded_seed_rebuilds_the_same_release_schedule(self):
        """The whole point: the verifier's own regeneration has to land on the
        world the run recorded, not on the one the shipped file describes."""
        run = _Run(self, None)
        used = list(run.env._balloon_release_at_step)

        findings, oracle = verifier.check_scenario_is_official(run.payload)

        self.assertTrue(
            all(finding.ok for finding in findings), [str(f) for f in findings]
        )
        _flights, schedule, _status = verifier._regenerate_balloon_flights(oracle)
        self.assertEqual(list(schedule), used)

    def test_a_seed_that_is_not_the_shipped_one_is_still_official(self):
        """What an arbitrary-seed round needs. Only the seed may differ."""
        findings, oracle = verifier.check_scenario_is_official(
            _submission(20260801, 20260801)
        )

        self.assertTrue(
            all(finding.ok for finding in findings), [str(f) for f in findings]
        )
        self.assertEqual(oracle["scenario"]["random_seed"], 20260801)

    def test_any_other_parameter_still_has_to_be_the_shipped_one(self):
        """The control for the exemption. Loosening the seed must not loosen
        the comparison it was carved out of."""
        submission = _submission(0, 0)
        parameters = submission["balloon_world_data"]["scenario_parameters"]
        parameters["balloon"]["radius"] = 99.0

        findings, _oracle = verifier.check_scenario_is_official(submission)

        self.assertFalse(all(finding.ok for finding in findings))
        self.assertTrue(any("radius" in finding.name for finding in findings))

    def test_the_two_places_the_seed_appears_have_to_agree(self):
        """Exempting the parameters from the comparison would otherwise leave a
        field a submission can say anything in."""
        findings, oracle = verifier.check_scenario_is_official(_submission(5, 6))

        self.assertIsNone(oracle)
        self.assertTrue(
            any("and the scenario parameters say" in f.detail for f in findings)
        )

    def test_a_null_in_the_parameters_is_what_a_random_run_looks_like(self):
        """The control for the check above. It has to allow the case it exists
        for, which is the run that drew its own seed."""
        findings, oracle = verifier.check_scenario_is_official(_submission(5, None))

        self.assertIsNotNone(oracle)
        self.assertTrue(
            all(finding.ok for finding in findings), [str(f) for f in findings]
        )

    def test_a_missing_recorded_seed_is_refused_with_a_reason(self):
        """A submission from a build that did not record it. Nothing physical
        can run, so it says so instead of rebuilding the wrong world."""
        findings, oracle = verifier.check_scenario_is_official(_submission(None, 0))

        self.assertIsNone(oracle)
        self.assertTrue(
            any(not f.ok and "cannot be rebuilt" in f.detail for f in findings),
            [str(f) for f in findings],
        )

    def test_a_boolean_is_not_a_seed(self):
        """`True` is an `int` by inheritance and would reset to seed 1."""
        _findings, oracle = verifier.check_scenario_is_official(_submission(True, 0))

        self.assertIsNone(oracle)

    def test_a_negative_seed_is_refused_rather_than_raising(self):
        """`SeedSequence` refuses negative entropy, so this would be a crash
        inside the regeneration rather than a finding."""
        _findings, oracle = verifier.check_scenario_is_official(_submission(-1, 0))

        self.assertIsNone(oracle)


if __name__ == "__main__":
    unittest.main()
