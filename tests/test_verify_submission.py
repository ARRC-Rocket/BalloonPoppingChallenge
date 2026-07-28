"""What the submission checker catches, and what it does not.

Every test here starts from a submission the environment actually produced and
then edits it, because a checker that passes a clean file proves nothing about
whether it would catch a dirty one. The edits are the ones somebody would
actually make: move a balloon onto the rocket's path, claim pops that did not
happen, raise the score.

Scenario 1 cut to a single balloon. Scenario 0 would be faster still and would
be useless: its balloons are static, so its flight array is identical at every
timestep and a comparison against it passes at any time offset at all. The
fixture asserts its balloon moves, so that cannot quietly become true again.
"""

import copy
import os
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_1_PARAMS = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_1_parameters.yaml"
)
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import yaml

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from verify_submission import (
        DEFAULT_TOLERANCE_METRES,
        _release_eligibility,
        _load_canonical_scenario,
        check_claimed_pops_are_reachable,
        check_internal_consistency,
        verify,
    )


def _build_submission(steps=40):
    """Run the real environment and package what it recorded.

    Built here rather than by calling ``pack_for_submission`` so the test does
    not write a file, reach the network for the integrity check, or depend on
    the container format. The fields below are the ones the checker reads.
    """
    parameters = yaml.safe_load(SCENARIO_1_PARAMS.read_text(encoding="utf-8"))
    parameters["balloon"]["num"] = 1
    # With one balloon the schedule is arange(1) * step, so it is released at
    # step 0. Asserted in the fixture test below rather than assumed, since a
    # short run past a balloon still on the ground would leave the pop checks
    # with nothing to look at.

    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    env.reset(seed=parameters["scenario"]["random_seed"])
    action = env.action_space.sample()
    action["launch"] = np.array(1, dtype=action["launch"].dtype)
    action["launch_inclination_heading"] = np.array([90.0, 0.0], dtype=np.float64)
    action["throttle"] = np.ones_like(action["throttle"])
    for key in ("tvc", "roll"):
        action[key] = np.zeros_like(action[key])
    for _ in range(steps):
        env.step(action)

    return {
        "leaderboard_info": {
            "team_name": "example",
            "agent_name": "a",
            "scenario_number": 1,
            "final_reward": int(env._popped_count),
        },
        "balloon_world_data": {
            "scenario_parameters": parameters,
            "trajectories": copy.deepcopy(env.trajectories),
            "balloon_release_at_step": list(env._balloon_release_at_step),
        },
    }


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheSubmissionChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = _build_submission()

    def setUp(self):
        self.submission = copy.deepcopy(self.clean)
        self.records = self.submission["balloon_world_data"]["trajectories"]
        # The fixture is scenario 1 cut to one balloon, so it is deliberately
        # not the scenario this repository ships and the real oracle rejects it
        # on sight. That rejection is the subject of its own class below; here
        # the reduced scenario stands in as the official one, so these tests
        # keep exercising what they are named for.
        #
        # A fresh copy each time, so a test that edits the submitted parameters
        # is editing only those.
        official = copy.deepcopy(
            self.clean["balloon_world_data"]["scenario_parameters"]
        )
        patcher = patch(
            "verify_submission._load_canonical_scenario",
            lambda number: copy.deepcopy(official),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def failures(self, submission=None):
        found = verify(submission or self.submission, DEFAULT_TOLERANCE_METRES)
        return [finding.name for finding in found if not finding.ok]

    def test_the_fixture_has_a_balloon_that_moves(self):
        """Otherwise every test below would pass at any time offset.

        This is the property scenario 0 does not have, and the reason these
        tests use scenario 1 despite it costing a Monte Carlo run.
        """
        positions = np.asarray(
            [record["balloon_states"] for record in self.records], dtype=float
        )[:, 0, :3]
        travelled = float(np.abs(np.diff(positions, axis=0)).max())

        self.assertGreater(travelled, 0.01, "the fixture balloon does not move")

    def test_an_untouched_submission_passes(self):
        self.assertEqual(self.failures(), [])

    def test_a_moved_balloon_is_caught(self):
        for shift, label in ((3.0, "three metres"), (0.01, "a centimetre")):
            with self.subTest(shift=label):
                self.submission = copy.deepcopy(self.clean)
                self.records = self.submission["balloon_world_data"]["trajectories"]
                for record in self.records:
                    record["balloon_states"][0][0] += shift

                self.assertIn("balloon trajectories", self.failures())

    def test_a_single_moved_timestep_is_caught(self):
        self.records[len(self.records) // 2]["balloon_states"][0][2] += 2.0

        self.assertIn("balloon trajectories", self.failures())

    def test_a_rewritten_score_is_caught(self):
        self.submission["leaderboard_info"]["final_reward"] = 87

        self.assertIn("score matches the record", self.failures())

    def test_a_status_going_backwards_is_caught(self):
        half = len(self.records) // 2
        self.records[half]["balloon_status"][0] = 1
        self.records[half + 1]["balloon_status"][0] = 0

        self.assertIn("balloon status only moves forward", self.failures())

    def test_a_fabricated_pop_is_caught(self):
        """The edit every other check here misses.

        Leave the positions alone, flip a released balloon to popped at a moment
        that looks legal, and move the score to match. Measured against an
        earlier version of the checker: twenty of these passed everything.

        Nothing here is out of order, so the cheap consistency checks have
        nothing to object to. What catches it is that the rocket never went
        near that balloon.
        """
        released = [
            index
            for index, status in enumerate(self.records[0]["balloon_status"])
            if status >= 1
        ]
        self.assertTrue(released, "the fixture released no balloons")

        for record in self.records[1:]:
            record["balloon_status"][released[0]] = 2
        self.submission["leaderboard_info"]["final_reward"] = 1

        failures = self.failures()
        self.assertNotIn("balloon status only moves forward", failures)
        self.assertNotIn("score matches the record", failures)
        self.assertNotIn("balloon trajectories", failures)
        self.assertIn("claimed pops are reachable", failures)

    def test_a_changed_seed_is_caught(self):
        """Caught by the parameter comparison now, which is the tighter answer.

        It used to be caught indirectly: regenerating with the edited seed
        produced balloons that did not match the recorded ones. That only worked
        because the recording was made with the real seed. A run genuinely
        carried out under a different seed agreed with itself and passed, which
        is the hole the shipped scenario closes.
        """
        parameters = self.submission["balloon_world_data"]["scenario_parameters"]
        parameters["scenario"]["random_seed"] = 99

        failures = self.failures()
        self.assertIn("scenario parameters are the shipped ones", failures)
        self.assertIn("parameter scenario.random_seed", failures)
        # And the balloons still compare clean, which is what says the
        # regeneration used the shipped scenario rather than this edited copy.
        # Regenerating from the submission would use seed 99, produce different
        # balloons from the recorded ones, and fail here too. Without this line
        # the two sources are indistinguishable to the suite.
        self.assertNotIn("balloon trajectories", failures)

    def test_the_canonical_world_is_rebuilt_once(self):
        """Rebuilding it runs the balloon Monte Carlo.

        It was being paid for twice per submission: once to work out release
        eligibility and once to compare the balloon trajectories. Asserted
        rather than left to the structure, since the next check to need the
        canonical world is one call away from making it three.
        """
        import verify_submission

        with patch.object(
            verify_submission,
            "_regenerate_balloon_flights",
            wraps=verify_submission._regenerate_balloon_flights,
        ) as regenerate:
            verify(self.submission, DEFAULT_TOLERANCE_METRES)

        self.assertEqual(regenerate.call_count, 1)

    def test_a_changed_release_schedule_is_caught(self):
        self.submission["balloon_world_data"]["balloon_release_at_step"][0] += 1

        self.assertIn("release schedule", self.failures())

    def test_a_missing_field_is_reported_rather_than_raised(self):
        del self.submission["balloon_world_data"]["trajectories"]

        self.assertEqual(self.failures(), ["structure"])


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheOracleIsTheShippedScenario(unittest.TestCase):
    """Nothing here patches the loader, because the loader is the subject.

    The checker used to regenerate the balloons from the ``scenario_parameters``
    inside the submission, which let the file under examination supply the
    answer it was checked against.
    """

    def test_the_loader_reads_this_repository(self):
        shipped, _ = load_scenario_parameters(0)

        self.assertEqual(_load_canonical_scenario(0), shipped)

    def test_a_scenario_this_repository_does_not_ship_has_no_oracle(self):
        self.assertIsNone(_load_canonical_scenario(7))

    def test_a_run_against_modified_parameters_is_caught_though_it_agrees_with_itself(
        self,
    ):
        """The whole point, and the fixture is already the right probe.

        It is scenario 1 with the balloon count changed, and everything in it
        was produced by the environment under exactly those parameters. So it is
        internally consistent: its balloons are the ones its own parameters
        imply, its statuses follow, its score matches. Against its own copy of
        the scenario it passed every check. Against the shipped one it does not.
        """
        submission = _build_submission()

        findings = verify(submission, DEFAULT_TOLERANCE_METRES)
        failures = [finding.name for finding in findings if not finding.ok]

        self.assertIn("scenario parameters are the shipped ones", failures)
        self.assertIn("parameter balloon.num", failures)

    def test_a_scenario_number_that_disagrees_with_the_parameters_is_caught(self):
        submission = _build_submission()
        submission["leaderboard_info"]["scenario_number"] = 0

        failures = [
            finding.name
            for finding in verify(submission, DEFAULT_TOLERANCE_METRES)
            if not finding.ok
        ]

        self.assertIn("scenario number", failures)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestWhenAPopIsTooEarly(unittest.TestCase):
    """ "2 never follows 0" reads like the rule and is not one.

    step() marks a balloon released, detects pops, and only then writes the
    record, so a balloon released and popped on the same step legitimately
    records 0 and then 2 with nothing in between. What actually holds is about
    the step number, not the shape of the sequence.
    """

    @staticmethod
    def _consistency(status_rows, release_at_step, eligibility="derive"):
        submission = {
            "leaderboard_info": {
                "final_reward": int((np.asarray(status_rows)[-1] == 2).sum())
            },
            "balloon_world_data": {
                "trajectories": [{"balloon_status": list(row)} for row in status_rows],
                "balloon_release_at_step": list(release_at_step),
            },
        }
        if isinstance(eligibility, str):
            status = np.asarray(status_rows, dtype=int)
            steps = np.arange(1, status.shape[0] + 1)[:, None]
            eligibility = steps >= np.asarray(release_at_step, dtype=int)[None, :]
        return [
            finding.name
            for finding in check_internal_consistency(submission, eligibility)
            if not finding.ok
        ]

    def test_a_scenario_that_starts_released_is_never_too_early(self):
        """Scenario 0 releases everything at reset and its schedule never fires.

        Measured: with the shipped seed balloon 0 is scheduled for step 400 and
        the committed baseline pops it at 370, so comparing against the schedule
        reported the repository's own run as popping before release.
        """
        eligible = np.ones((3, 1), dtype=bool)

        failures = self._consistency([[1], [2], [2]], [400], eligibility=eligible)

        self.assertNotIn("no balloon is popped before release", failures)

    def test_released_and_popped_on_the_same_step_is_allowed(self):
        # Row 0 holds step 1, so a balloon released on step 1 may read 2 there.
        failures = self._consistency([[0], [2], [2]], release_at_step=[1])

        self.assertNotIn("no balloon is popped before release", failures)

    def test_without_a_release_rule_the_check_reports_that_it_did_not_run(self):
        """An unevaluated check must not look like a passed one.

        It used to leave ``early`` empty and report "every pop is on or after
        the balloon's release step", which is an affirmative line about a
        comparison that never happened.
        """
        failures = self._consistency([[2], [2]], [0], eligibility=None)  # really None

        self.assertIn("no balloon is popped before release", failures)

    def test_popped_before_the_release_step_is_caught(self):
        # Row 0 is step 1 again, and this balloon is not released until step 50.
        failures = self._consistency([[2], [2], [2]], release_at_step=[50])

        self.assertIn("no balloon is popped before release", failures)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheIntervalsThePopCheckLooksAt(unittest.TestCase):
    """Two boundaries the environment sweeps and the checker used to skip.

    Both could only produce a false accusation, which is the expensive direction
    for a checker that decides whether somebody cheated. Driven directly rather
    than through ``verify``, because ``verify`` only reaches the pop check once
    everything else has passed, and these cases are about the pop check alone.

    The submissions built here carry no ``scenario_parameters`` on purpose. The
    radius and the pad elevation have to come from the shipped scenario, so
    reading either of them off the submission raises a KeyError here rather than
    letting the file being checked widen the check that judges it.
    """

    ELEVATION = 20.0
    RADIUS = 1.5
    CANONICAL = {"balloon": {"radius": RADIUS}, "environment": {"elevation": ELEVATION}}

    @staticmethod
    def _submission(rocket_rows, status_rows):
        return {
            "balloon_world_data": {
                "trajectories": [
                    {
                        "rocket_states": list(rocket) + [0.0] * 10,
                        "balloon_status": list(status),
                    }
                    for rocket, status in zip(rocket_rows, status_rows)
                ]
            }
        }

    def _closest(self, rocket_rows, status_rows, balloon_positions, eligible=None):
        status = np.asarray(status_rows, dtype=int)
        if eligible is None:
            # These cases are about the interval boundaries, not about release
            # timing, so every balloon is eligible throughout.
            eligible = np.ones(status.shape, dtype=bool)
        findings = check_claimed_pops_are_reachable(
            self._submission(rocket_rows, status_rows),
            np.asarray(balloon_positions, dtype=float),
            self.CANONICAL,
            eligible,
        )
        return [finding for finding in findings if not finding.ok]

    def test_a_pop_over_the_launch_interval_is_not_an_accusation(self):
        """The environment sweeps that interval from the pad, not from a record.

        On the launch step the rocket has not been stepped, so its recorded
        state is NaN and the interval into the first stepped position has only
        one real end. An interval needing two dropped it, and a balloon popped
        against the pad looked unreachable.
        """
        nan = [float("nan")] * 3
        rocket = [nan, [0.0, 0.0, 60.0], [0.0, 0.0, 140.0]]
        status = [[1], [2], [2]]
        # Half a metre above the pad the whole time, so the only interval that
        # can reach it is the one starting there.
        balloons = [[[0.0, 0.0, self.ELEVATION + 0.5]] for _ in range(3)]

        self.assertEqual(self._closest(rocket, status, balloons), [])

    def test_the_launch_pad_does_not_excuse_an_unreachable_pop(self):
        """Or the test above would pass by reaching everything.

        Same shape, with the balloon parked far from both the pad and the path.
        """
        nan = [float("nan")] * 3
        rocket = [nan, [0.0, 0.0, 60.0], [0.0, 0.0, 140.0]]
        status = [[1], [2], [2]]
        balloons = [[[500.0, 500.0, self.ELEVATION + 0.5]] for _ in range(3)]

        self.assertNotEqual(self._closest(rocket, status, balloons), [])

    def test_a_balloon_released_and_popped_on_the_same_step_is_not_an_accusation(self):
        """step() releases, then detects pops, then writes the record.

        So the row before shows 0 and the row itself shows 2, and reading the
        status at the start of the interval said the balloon was still on the
        ground and skipped the interval it was popped over.
        """
        rocket = [
            [0.0, 0.0, self.ELEVATION],
            [40.0, 0.0, self.ELEVATION],
            [80.0, 0.0, self.ELEVATION],
        ]
        status = [[0], [2], [2]]
        # Sitting on the segment the rocket flies between rows 0 and 1.
        balloons = [[[20.0, 0.0, self.ELEVATION]] for _ in range(3)]
        # Row 0 is step 1 and the balloon is released on step 2, so it becomes
        # eligible at row 1, which is the end of the interval it is popped over.
        eligible = np.array([[False], [True], [True]])

        self.assertEqual(self._closest(rocket, status, balloons, eligible), [])

    def test_a_status_forged_released_before_the_scenario_allows_is_ignored(self):
        """The release mask comes from the scenario, not from the submission.

        Reading the submitted status let a file claim a balloon was released
        early, point at a rocket pass from before the real release as its
        closest approach, and flip to popped on the official step. Both the
        timing check and this one passed.
        """
        rocket = [
            [0.0, 0.0, self.ELEVATION],
            [40.0, 0.0, self.ELEVATION],
            [4000.0, 0.0, self.ELEVATION],
        ]
        # The file says released from the first row and popped on the last.
        status = [[1], [1], [2]]
        balloons = [[[20.0, 0.0, self.ELEVATION]] for _ in range(3)]
        # The scenario says it is not released until the last row.
        eligible = np.array([[False], [False], [True]])

        self.assertNotEqual(self._closest(rocket, status, balloons, eligible), [])


class TestTheReleaseRuleItself(unittest.TestCase):
    """The rule, straight from the regenerated facts.

    Nothing else in this file reaches it. Scenario 0 starts every balloon
    released, and the reduced scenario-1 fixture has one balloon and therefore a
    release step of zero, so both produce an all-True mask; the forged-release
    test supplies its own. Measured: returning ``ones(...)`` from this function,
    and moving its comparison off by one, both left every other test here green,
    which is exactly the bypass this branch exists to close.
    """

    def test_a_grounded_balloon_is_not_eligible_until_its_step(self):
        # Row k is step k + 1, so a balloon released on step 3 is eligible from
        # row 2 onwards.
        eligibility = _release_eligibility(np.array([3]), np.array([0]), steps=4)

        np.testing.assert_array_equal(eligibility, [[False], [False], [True], [True]])

    def test_a_balloon_that_starts_released_is_eligible_throughout(self):
        """Scenario 0. Its schedule still says 400 and it never fires."""
        eligibility = _release_eligibility(np.array([400]), np.array([1]), steps=3)

        np.testing.assert_array_equal(eligibility, [[True], [True], [True]])

    def test_the_two_rules_combine_per_balloon(self):
        eligibility = _release_eligibility(
            np.array([2, 400]), np.array([0, 1]), steps=3
        )

        np.testing.assert_array_equal(
            eligibility, [[False, True], [True, True], [True, True]]
        )


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
@unittest.skipUnless(
    os.environ.get("BPC_RUN_SLOW_TESTS", "0").strip().lower() in ("1", "true", "yes"),
    "set BPC_RUN_SLOW_TESTS=1 to run a full scenario",
)
class TestARealScenario0SubmissionPasses(unittest.TestCase):
    """The check this file did not have, and the one that mattered.

    Every other test here builds a submission by hand, from scenario 1 cut to
    one balloon. None of them ran the shipped scenario 0 end to end, and that is
    what let the release-eligibility bug ship: scenario 0 starts every balloon
    released and its schedule never fires, so comparing pops against the
    schedule reported the repository's own run as popping balloon 0 at step 370
    when the schedule said 400.

    An honest official submission passing is the cheapest possible statement
    that the checker is fit to judge one.
    """

    def test_the_shipped_scenario_0_run_is_not_accused_of_anything(self):
        from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        parameters, given = load_scenario_parameters(0)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        agent = AttitudeRateControlAgent(
            given, rate_targets=[0.0, 0.0, 0.0], launch_time=1
        )
        observation, _ = env.reset(seed=parameters["scenario"]["random_seed"])
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _, terminated, truncated, _ = env.step(
                agent.get_action(observation)
            )

        submission = {
            "leaderboard_info": {
                "team_name": "official",
                "agent_name": "AttitudeRateControlAgent",
                "scenario_number": 0,
                "final_reward": int(env._popped_count),
            },
            "balloon_world_data": {
                "scenario_parameters": parameters,
                "trajectories": copy.deepcopy(env.trajectories),
                "balloon_release_at_step": list(env._balloon_release_at_step),
            },
        }

        findings = verify(submission, DEFAULT_TOLERANCE_METRES)
        failures = [f"{f.name}: {f.detail}" for f in findings if not f.ok]

        self.assertEqual(failures, [])
        self.assertGreater(env._popped_count, 0, "this run should pop something")


if __name__ == "__main__":
    unittest.main()
