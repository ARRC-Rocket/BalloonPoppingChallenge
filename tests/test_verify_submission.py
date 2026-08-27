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
    from verify_submission import (
        CUMULATIVE_DRIFT_TOLERANCE_METRES,
        DEFAULT_TOLERANCE_METRES,
        VELOCITY_CONSISTENCY_TOLERANCE,
        _load_canonical_scenario,
        _release_eligibility,
        check_claimed_pops_are_reachable,
        check_internal_consistency,
        check_the_rocket_path_is_a_trajectory,
        verify,
    )

    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def _build_submission(steps=40, balloons=1):
    """Run the real environment and package what it recorded.

    Built here rather than by calling ``pack_for_submission`` so the test does
    not write a file, reach the network for the integrity check, or depend on
    the container format. The fields below are the ones the checker reads.

    ``balloons`` exists because one gives a schedule of ``[0]``, eligible from
    the first row, so the release rule cannot be seen. Two gives ``[0, 50]``.
    """
    parameters = yaml.safe_load(SCENARIO_1_PARAMS.read_text(encoding="utf-8"))
    parameters["balloon"]["num"] = balloons
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
        # What `build_submission_payload` writes, so the fixture is the shape the
        # checker is pointed at rather than a subset of it.
        "format_version": 1,
        "leaderboard_info": {
            "team_name": "example",
            "agent_name": "a",
            "scenario_number": 1,
            "final_reward": int(env._popped_count),
            "random_seed": env.np_random_seed,
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

    def test_a_seed_the_run_did_not_use_is_caught(self):
        """The seed is an input to the oracle now, so editing it in the file and
        leaving the balloons alone regenerates a different world and compares the
        recorded data against that.

        It used to be caught by the parameter comparison, which refused any seed
        but the shipped one. That also refused every honest run on any other
        seed, which is what an arbitrary-seed round is made of. The property that
        matters survives the change: a submission still has to hold the balloons
        the seed it names produces.
        """
        # Both places, so this is a submission claiming a world rather than one
        # whose two copies of the seed disagree. That is its own finding.
        self.submission["leaderboard_info"]["random_seed"] = 99
        parameters = self.submission["balloon_world_data"]["scenario_parameters"]
        parameters["scenario"]["random_seed"] = 99

        failures = self.failures()
        self.assertNotIn("scenario parameters are the shipped ones", failures)
        self.assertNotIn("the seed the run used", failures)
        self.assertIn("balloon trajectories", failures)

    def test_the_canonical_world_is_rebuilt_once(self):
        """Rebuilding it runs the balloon Monte Carlo.

        It was paid for twice per submission, once for release eligibility and
        once for the trajectories. The next check to need it makes it three.
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

        With the shipped seed balloon 0 is scheduled for step 400 and the
        baseline pops it at 370, which the schedule alone calls too early.
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
        the balloon's release step" about a comparison that never happened.
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

        Reading the submitted status let a file claim an early release, point at
        a rocket pass from before the real one, and pop on the official step.
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


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheReleaseRuleItself(unittest.TestCase):
    """The rule, straight from the regenerated facts.

    Nothing else here reaches it, both fixtures give an all-True mask. Returning
    ``ones(...)`` or moving the comparison off by one left every other test green.
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

    Every other test builds its submission by hand from scenario 1 cut to one
    balloon. None ran the shipped scenario 0 end to end, which let the bug ship.
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
            "format_version": 1,
            "leaderboard_info": {
                "team_name": "official",
                "agent_name": "AttitudeRateControlAgent",
                "scenario_number": 0,
                "final_reward": int(env._popped_count),
                "random_seed": env.np_random_seed,
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


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheReleaseRuleReachesTheVerdict(unittest.TestCase):
    """The rule was correct and wired to nothing any test could see.

    Replacing it with ``ones_like`` in ``verify()`` left all 27 tests green and
    accepted a forged score. Two balloons, schedule ``[0, 50]``, tells them apart.
    """

    RELEASE_STEP = 50

    @classmethod
    def setUpClass(cls):
        cls.clean = _build_submission(steps=40, balloons=2)

    def setUp(self):
        self.submission = copy.deepcopy(self.clean)
        self.records = self.submission["balloon_world_data"]["trajectories"]
        official = copy.deepcopy(
            self.clean["balloon_world_data"]["scenario_parameters"]
        )
        patcher = patch(
            "verify_submission._load_canonical_scenario",
            lambda number: copy.deepcopy(official),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def failures(self):
        return [
            finding.name
            for finding in verify(self.submission, DEFAULT_TOLERANCE_METRES)
            if not finding.ok
        ]

    def test_the_fixture_has_a_balloon_that_is_not_released_yet(self):
        """Or the two tests below would hold with the rule deleted.

        The run is 40 steps and balloon 1 is released at 50, so it is on the
        ground throughout and a claim against it is refusable on its own.
        """
        schedule = self.submission["balloon_world_data"]["balloon_release_at_step"]

        self.assertEqual(list(schedule), [0, self.RELEASE_STEP])
        self.assertGreater(self.RELEASE_STEP, len(self.records))

    def test_an_honest_two_balloon_submission_still_passes(self):
        """The half that stops the next one passing by refusing everything."""
        self.assertEqual(self.failures(), [])

    def test_a_pop_claimed_before_release_is_refused_through_verify(self):
        """The forgery the rule exists to stop, driven through the entry point.

        The status is forged to released and then popped. Only the scenario
        knows balloon 1 is on the ground, so only its mask can refuse this.
        """
        for record in self.records[1:]:
            record["balloon_status"][1] = 2
        self.submission["leaderboard_info"]["final_reward"] = 1

        self.assertIn("no balloon is popped before release", self.failures())

    def test_a_status_matrix_of_the_wrong_width_is_reported_not_raised(self):
        """The guard that keeps a bad file from ending the run.

        Named, not just "something failed": the shape check below and the
        release rule can both refuse this, and which one did is the difference
        between a report a competitor can act on and a misleading one.
        """
        for record in self.records:
            record["balloon_status"] = list(record["balloon_status"]) + [0]

        self.assertIn("balloon_status", self.failures())

    def test_a_scenario_that_disagrees_with_its_own_balloon_count_is_reported(self):
        """The other way the mask and the status can differ in width, and the
        only one the shape check cannot see: it takes its count from the
        scenario file, and this is the scenario file disagreeing with the
        environment it configures. Unguarded, numpy raises out of ``verify``."""
        import verify_submission

        real = verify_submission._regenerate_balloon_flights

        def one_balloon_short(parameters):
            flights, release_at_step, initial_status = real(parameters)
            return flights[:-1], release_at_step[:-1], initial_status[:-1]

        with patch.object(
            verify_submission, "_regenerate_balloon_flights", one_balloon_short
        ):
            failures = self.failures()

        self.assertIn("balloon status shape", failures)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestWhatABadShapeCostsBeforeItIsRefused(unittest.TestCase):
    """Rebuilding the world runs the balloon Monte Carlo, a hundred flights for
    scenario 1. A file that could be refused for nothing should not pay it."""

    @classmethod
    def setUpClass(cls):
        cls.clean = _build_submission(steps=40, balloons=2)

    def setUp(self):
        self.submission = copy.deepcopy(self.clean)
        self.records = self.submission["balloon_world_data"]["trajectories"]
        official = copy.deepcopy(
            self.clean["balloon_world_data"]["scenario_parameters"]
        )
        patcher = patch(
            "verify_submission._load_canonical_scenario",
            lambda number: copy.deepcopy(official),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def refused_without_regenerating(self):
        """Run ``verify`` with the Monte Carlo booby trapped, and return the
        failures. Reaching it at all is the failure being tested for."""
        with patch(
            "verify_submission._regenerate_balloon_flights",
            side_effect=AssertionError("regenerated before checking the shape"),
        ):
            findings = verify(self.submission, DEFAULT_TOLERANCE_METRES)
        return [finding.name for finding in findings if not finding.ok]

    def test_an_honest_submission_is_the_one_that_pays_for_it(self):
        """The half that stops every test here passing over a checker that
        refuses everything before it gets as far as the physics."""
        with self.assertRaises(AssertionError):
            self.refused_without_regenerating()

    def test_a_ragged_status_row_is_refused_first(self):
        """``np.asarray`` raises on this rather than giving an array, so
        unchecked it comes out of ``verify`` as a traceback."""
        self.records[3]["balloon_status"] = [0]

        self.assertIn("balloon_status", self.refused_without_regenerating())

    def test_a_states_array_of_the_wrong_width_is_refused_first(self):
        for record in self.records:
            record["balloon_states"] = [row[:5] for row in record["balloon_states"]]

        self.assertIn("balloon_states", self.refused_without_regenerating())

    def test_a_record_missing_a_field_is_refused_first(self):
        del self.records[2]["rocket_states"]

        self.assertIn("rocket_states", self.refused_without_regenerating())

    def test_a_record_that_is_not_a_mapping_is_refused_first(self):
        self.records[1] = [0.0, 1.0]

        self.assertIn("record structure", self.refused_without_regenerating())

    def test_a_number_too_large_for_a_float_is_reported_not_raised(self):
        """A submission is a file somebody else writes, so the shape check has
        to answer rather than raise. `10**400` came out of `verify()` as an
        OverflowError traceback."""
        self.records[0]["balloon_states"][0][0] = 10**400

        self.assertIn("balloon_states", self.refused_without_regenerating())

    def test_the_wrong_number_of_balloons_is_refused_first(self):
        """The count comes from the canonical scenario, so this needs no
        Monte Carlo to answer either."""
        for record in self.records:
            record["balloon_status"] = list(record["balloon_status"]) + [0]

        self.assertIn("balloon_status", self.refused_without_regenerating())


class TestOneBadFileDoesNotStopTheBatch(unittest.TestCase):
    """``main()`` caught a failed read and not a failed check, and checking is
    the half that touches the arrays the submission controls."""

    def test_a_verify_that_raises_costs_that_file_only(self):
        import verify_submission

        submissions = [{"leaderboard_info": {}}, {"leaderboard_info": {}}]
        seen = []

        def explode_once(submission, tolerance):
            seen.append(submission)
            if len(seen) == 1:
                raise ValueError("ragged")
            return [verify_submission.Finding("checked", True, "fine")]

        with (
            patch.object(
                verify_submission,
                "load_submission",
                lambda path: submissions[len(seen)],
            ),
            patch.object(verify_submission, "verify", explode_once),
        ):
            status = verify_submission.main(["first.pkl", "second.pkl"])

        self.assertEqual(len(seen), 2, "the second file was never checked")
        self.assertEqual(status, 1)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheRocketPathHasToBeATrajectory(unittest.TestCase):
    """The path is the competitor's claim, so ask what faking it costs.

    Driven through the check directly, using a real run as the starting point,
    because the point is what happens when that run is edited.
    """

    ELEVATION = 20.0

    @classmethod
    def setUpClass(cls):
        cls.clean = _build_submission()
        cls.canonical = {
            "environment": {"elevation": cls.ELEVATION},
            "simulation": {"time_step": 0.01},
        }

    def setUp(self):
        self.submission = copy.deepcopy(self.clean)
        self.records = self.submission["balloon_world_data"]["trajectories"]

    def failures(self):
        return [
            finding.name
            for finding in check_the_rocket_path_is_a_trajectory(
                self.submission, self.canonical
            )
            if not finding.ok
        ]

    def _flown(self):
        return [
            record
            for record in self.records
            if np.isfinite(record["rocket_states"][:3]).all()
        ]

    def test_a_real_run_passes(self):
        """The half that stops the rest passing by refusing everything."""
        self.assertEqual(self.failures(), [])

    def test_dragging_the_path_sideways_is_caught(self):
        """The edit that would put the rocket on a balloon.

        Positions move and the velocities beside them do not, which is the
        whole point: they came out of one integration and no longer agree.
        """
        for offset, record in enumerate(self._flown()):
            record["rocket_states"][0] += 0.5 * offset

        self.assertIn("recorded velocity matches the path", self.failures())

    def test_a_single_displaced_step_is_caught(self):
        flown = self._flown()
        flown[len(flown) // 2]["rocket_states"][1] += 2.0

        self.assertIn("recorded velocity matches the path", self.failures())

    def test_a_displacement_below_the_tolerance_is_not_an_accusation(self):
        """The bound is measured, so it has to leave an honest run alone.

        Half a tolerance is seventeen times the 0.002923 m/s a complete honest
        run shows on either scenario.
        """
        flown = self._flown()
        step = 0.01
        nudge = 0.5 * VELOCITY_CONSISTENCY_TOLERANCE * step
        flown[len(flown) // 2]["rocket_states"][1] += nudge

        self.assertNotIn("recorded velocity matches the path", self.failures())

    def test_an_offset_on_every_other_row_is_caught(self):
        """The hole a central difference cannot see.

        ``(p[k+1] - p[k-1]) / (2 dt)`` only compares rows of the same parity, so
        1000 m added to every odd row cancelled exactly and every finding passed.
        """
        for index, record in enumerate(self._flown()):
            if index % 2:
                record["rocket_states"][0] += 1000.0

        self.assertIn("recorded velocity matches the path", self.failures())

    def test_an_attitude_that_is_not_a_rotation_is_caught(self):
        for record in self._flown():
            record["rocket_states"][6] *= 1.5

        self.assertIn("attitude is a rotation", self.failures())

    def test_a_flight_that_does_not_start_on_the_pad_is_caught(self):
        for record in self._flown():
            record["rocket_states"][2] += 500.0

        self.assertIn("the flight starts on the pad", self.failures())

    # How far into the last step a real flight gets before the ground stops it,
    # as a fraction of what its recorded speed would have carried it. Measured
    # on complete runs: 0.675 on scenario 0 and 0.641 on scenario 1.
    IMPACT_STEP_FRACTION = 0.675
    FROZEN_ROWS = 2

    def _with_impact_tail(
        self, fraction=IMPACT_STEP_FRACTION, frozen=FROZEN_ROWS, jump_metres=0.0
    ):
        """The rows a real landing leaves behind; the 40 step fixture has none.

        The last step is partial, because the integrator stops at the ground
        partway through it, and the state then repeats while still being recorded.
        """
        step = float(self.canonical["simulation"]["time_step"])
        last = copy.deepcopy(self.records[-1])
        velocity = np.asarray(last["rocket_states"][3:6], dtype=float)
        impact = copy.deepcopy(last)
        for axis in range(3):
            impact["rocket_states"][axis] += fraction * velocity[axis] * step
        impact["rocket_states"][0] += jump_metres
        for _ in range(1 + frozen):
            self.records.append(copy.deepcopy(impact))

    def test_a_flight_that_landed_is_not_an_accusation(self):
        """The case that shipped broken, and the reason for the whole rework.

        The previous implementation allowed 1 m/s and reported 22.66 m/s on an
        honest scenario 0 and 25.67 on scenario 1, all of it the partial last step.
        """
        self._with_impact_tail()

        self.assertEqual(self.failures(), [])

    def test_the_tail_is_trimmed_rather_than_tolerated(self):
        """The other half: the trim must not be a tolerance wide enough to
        swallow the artefact, or it would swallow a forged path too."""
        self._with_impact_tail()
        flown = self._flown()
        flown[len(flown) // 2]["rocket_states"][1] += 2.0

        self.assertIn("recorded velocity matches the path", self.failures())

    def test_the_last_step_may_be_short_but_not_long(self):
        """The last interval is excluded from the two-sided check, so it needs
        its own. Landing short is honest; covering more ground than the step
        before it is not. Five metres is more than three balloon radii.
        """
        self._with_impact_tail(jump_metres=5.0)

        self.assertIn("the last step is not a jump", self.failures())

    def test_the_last_step_is_bounded_by_positions_not_by_a_claimed_speed(self):
        """Sizing that bound from the recorded velocity hands it to the forger.

        ``velocity[-1]`` is the one row no two-sided comparison constrains, so a
        10000 m/s claim there used to buy 100 m of allowance for this 5 m jump.
        """
        self._with_impact_tail(jump_metres=5.0)
        for record in self.records[-3:]:
            record["rocket_states"][3:6] = [10000.0, 0.0, 0.0]

        self.assertIn("the last step is not a jump", self.failures())

    def test_a_jump_on_the_last_interior_step_is_caught(self):
        """The seam between the two checks, which is where an off-by-one hides.

        Narrowing the two-sided check by one interval left every other test here
        passing: the one-sided bound is sized from the interval that went unchecked.
        """
        self._with_impact_tail()
        self.records[-4]["rocket_states"][0] += 5.0

        self.assertIn("recorded velocity matches the path", self.failures())

    def test_a_ragged_row_is_reported_rather_than_raised(self):
        """The guard below the array was written for this and never saw it.

        ``np.asarray(..., dtype=float)`` raises on a ragged list one line before
        the shape check runs, so ``verify()`` used to see an unhandled ValueError.
        """
        for row, bad in enumerate(([1.0, 2.0], "not a row", 7.0)):
            with self.subTest(shape=repr(bad)):
                self.submission = copy.deepcopy(self.clean)
                self.records = self.submission["balloon_world_data"]["trajectories"]
                self.records[len(self.records) // 2]["rocket_states"] = bad

                self.assertIn("rocket state shape", self.failures())

    def test_a_jump_on_the_first_interval_is_caught(self):
        """The other end of the interior, and the other place an off-by-one hides.

        The seam at the last interval has its own test; narrowing the two-sided
        range at the front instead left every one of them passing.
        """
        self._flown()[0]["rocket_states"][0] += 5.0

        self.assertIn("recorded velocity matches the path", self.failures())

    def test_a_frozen_tail_of_an_odd_length_is_trimmed_exactly(self):
        """Every fixture here had an even tail, so trimming two at a time passed.

        Asserted on the count, since both trims leave something that looks like a
        flight. Under-trimming leaves the 139.6 m/s impact velocity in the series.
        """
        self._with_impact_tail(frozen=3)

        detail = next(
            finding.detail
            for finding in check_the_rocket_path_is_a_trajectory(
                self.submission, self.canonical
            )
            if finding.name == "the flight ends where it stops"
        )

        self.assertEqual(self.failures(), [])
        self.assertEqual(int(detail.split()[0]), 3)

    def test_the_last_step_bound_stays_near_the_step_before_it(self):
        """The multiplier, not just the slack beside it.

        Widening it to 100 left every other test passing while a jump of a
        hundred metres went through.
        """
        previous = float(
            np.linalg.norm(
                np.asarray(self._flown()[-1]["rocket_states"][:3], dtype=float)
                - np.asarray(self._flown()[-2]["rocket_states"][:3], dtype=float)
            )
        )
        self._with_impact_tail(jump_metres=3.0 * previous + 0.2)

        self.assertIn("the last step is not a jump", self.failures())

    def test_a_two_row_teleport_is_refused(self):
        """The shortest forgery there is: leave the pad, arrive at a balloon.
        Two rows have no interior, and reporting that as "too few to check" was
        a fail-open. One interval is still an interval and can be judged."""
        pad = list(
            self.records[len(self.records) - len(self._flown())]["rocket_states"]
        )
        target = list(pad)
        target[0] += 400.0
        self.records[:] = [{"rocket_states": pad}, {"rocket_states": target}]

        self.assertNotEqual(self.failures(), [])

    def test_teleporting_and_then_freezing_is_caught(self):
        """Freeze the tail and the trim throws away the evidence.

        Jump to a balloon, record zero velocity, repeat that row to the end, and
        the jump becomes the last surviving interval.
        """
        flown = self._flown()
        frozen = copy.deepcopy(flown[len(flown) // 2])
        frozen["rocket_states"][0] += 500.0
        frozen["rocket_states"][3:6] = [0.0, 0.0, 0.0]
        for record in flown[len(flown) // 2 :]:
            record["rocket_states"] = list(frozen["rocket_states"])

        self.assertNotEqual(self.failures(), [])

    def test_a_long_frozen_tail_is_refused(self):
        """A rocket parked in a good spot pops every balloon that drifts onto
        it, and the trim would hand those rows back unchecked. Measured on both
        scenarios: exactly two repeated positions."""
        self._with_impact_tail(frozen=400)

        self.assertIn("the flight ends where it stops", self.failures())

    def test_a_flight_too_short_to_check_is_not_a_pass(self):
        """Insufficient evidence is not evidence of innocence.

        The previous implementation reported "too few flown steps to
        differentiate" as a passing finding.
        """
        del self.records[len(self.records) - len(self._flown()) + 2 :]

        self.assertIn("recorded velocity matches the path", self.failures())

    def test_a_nan_in_a_column_nothing_reads_does_not_erase_the_flight(self):
        """The way through the whole of this check.

        Filtering to entirely finite rows meant one NaN in a body rate, a column
        nothing here reads, hid a path dragged sideways by a factor of 37.
        """
        for record in self._flown():
            record["rocket_states"][10] = float("nan")
            record["rocket_states"][0] *= 37.0

        self.assertIn("rocket state shape", self.failures())

    def test_a_partly_non_finite_row_is_refused(self):
        """Neither of the two shapes a run produces."""
        flown = self._flown()
        flown[len(flown) // 2]["rocket_states"][3] = float("nan")

        self.assertIn("rocket state shape", self.failures())

    def test_a_pre_launch_row_in_the_middle_of_the_flight_is_refused(self):
        """A whole row of NaN reads as pre-launch, a shape a run does make, so
        the partial-row check above does not see it. Dropping the row would
        splice samples 0.02 s apart into a series differentiated at 0.01 s.
        """
        records = self.records
        flown = [
            index
            for index, record in enumerate(records)
            if np.isfinite(record["rocket_states"][:3]).all()
        ]
        middle = flown[len(flown) // 2]
        records[middle]["rocket_states"] = [float("nan")] * len(
            records[middle]["rocket_states"]
        )

        # The gap check's own name, not the shared "rocket path" this asserted
        # before: a NaN row fails the velocity check too, so the old assertion
        # held with the gap check disabled.
        self.assertIn("the flight has no gaps", self.failures())

    def test_a_state_that_is_not_thirteen_wide_is_refused(self):
        for record in self.records:
            record["rocket_states"] = list(record["rocket_states"])[:11]

        self.assertIn("rocket state shape", self.failures())

    def test_a_quaternion_in_the_frozen_tail_is_still_checked(self):
        """Trimming the tail used to remove it from this check as well.

        The tail only needs excluding from the differentiation, so a repeated
        position carrying something that is not a rotation went unexamined.
        """
        self._with_impact_tail()
        for record in self.records[-3:]:
            record["rocket_states"][6] = 1.5

        self.assertIn("attitude is a rotation", self.failures())

    def test_a_run_that_never_launched_is_not_an_accusation(self):
        for record in self.records:
            record["rocket_states"] = [float("nan")] * len(record["rocket_states"])

        self.assertEqual(self.failures(), [])

    def test_verify_runs_this_check(self):
        """The checks above call the function directly, which says nothing
        about whether the tool does.

        Measured: unwiring it from ``verify()`` left every one of them passing.
        """
        for offset, record in enumerate(self._flown()):
            record["rocket_states"][0] += 0.5 * offset
        official = copy.deepcopy(
            self.clean["balloon_world_data"]["scenario_parameters"]
        )
        with patch(
            "verify_submission._load_canonical_scenario",
            lambda number: copy.deepcopy(official),
        ):
            findings = verify(self.submission, DEFAULT_TOLERANCE_METRES)

        self.assertIn(
            "recorded velocity matches the path",
            [finding.name for finding in findings if not finding.ok],
        )


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestACompleteFlightIsAccepted(unittest.TestCase):
    """One test on the artifact competitors actually upload.

    Everything above edits a 40 step run that never reaches the ground, and that
    is where this check went wrong. Scenario 0 runs to termination, ten seconds.
    """

    @classmethod
    def setUpClass(cls):
        parameters, given = load_scenario_parameters(0)
        cls.canonical = parameters
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
        cls.trajectories = copy.deepcopy(env.trajectories)

    def submission(self):
        return {
            "balloon_world_data": {"trajectories": copy.deepcopy(self.trajectories)}
        }

    def findings(self, submission):
        return check_the_rocket_path_is_a_trajectory(submission, self.canonical)

    def test_a_complete_honest_flight_is_accepted(self):
        """Measured against the previous implementation: 22.66 m/s of
        disagreement on this exact run, against a 1 m/s bound."""
        failed = [
            f"{finding.name}: {finding.detail}"
            for finding in self.findings(self.submission())
            if not finding.ok
        ]

        self.assertEqual(failed, [])

    def _worst_disagreement(self):
        detail = next(
            finding.detail
            for finding in self.findings(self.submission())
            if finding.name == "recorded velocity matches the path"
        )
        return float(detail.split()[2])

    def test_the_honest_run_sits_well_under_the_bound(self):
        """Measured: 0.002923 m/s, against a bound of 0.05."""
        self.assertLess(
            self._worst_disagreement(), 0.5 * VELOCITY_CONSISTENCY_TOLERANCE
        )

    def test_the_bound_stays_close_to_what_was_measured(self):
        """The other direction, and the one nothing else here holds.

        Every test here passes with the bound at the 1 m/s it shipped with, and
        at 1 m/s a drag of a centimetre per step is a balloon radius every 150.
        """
        self.assertLess(
            VELOCITY_CONSISTENCY_TOLERANCE, 100.0 * self._worst_disagreement()
        )

    def test_an_offset_on_every_other_row_is_caught_on_a_real_flight(self):
        """The nullspace, demonstrated on the artifact rather than a fixture.

        Measured on this run against the central difference: every finding
        passed with all 2954 odd rows moved 1000 m east.
        """
        submission = self.submission()
        records = submission["balloon_world_data"]["trajectories"]
        flown = [
            record
            for record in records
            if np.isfinite(record["rocket_states"][:3]).all()
        ]
        for index, record in enumerate(flown):
            if index % 2:
                record["rocket_states"][0] += 1000.0

        self.assertIn(
            "recorded velocity matches the path",
            [finding.name for finding in self.findings(submission) if not finding.ok],
        )

    def test_a_ramp_under_the_rate_bound_is_still_caught(self):
        """The bound above is a rate; what is being defended is a distance.

        Holding just under the per-step bound all flight buys 2.778 m of lateral
        drift, near two balloon radii. Measured here at a twentieth of that bound.
        """
        submission = self.submission()
        records = submission["balloon_world_data"]["trajectories"]
        flown = [
            record
            for record in records
            if np.isfinite(record["rocket_states"][:3]).all()
        ]
        position = np.asarray(
            [record["rocket_states"][:3] for record in flown], dtype=float
        )
        end = len(position)
        while end > 1 and np.array_equal(position[end - 1], position[end - 2]):
            end -= 1
        step = 0.05 * VELOCITY_CONSISTENCY_TOLERANCE * 0.01
        for index, record in enumerate(flown):
            record["rocket_states"][0] += min(index, end - 1) * step

        findings = {finding.name: finding.ok for finding in self.findings(submission)}

        self.assertTrue(
            findings["recorded velocity matches the path"],
            "the per-step check was expected to have nothing to say here",
        )
        self.assertFalse(findings["the path does not drift away from its velocity"])

    def test_the_honest_run_does_not_drift(self):
        """The half that stops the above being satisfied by refusing everything.

        Integration error alternates in sign rather than accumulating: the
        largest running total is 1.04e-04 m on scenario 0, 8.55e-05 m on 1.
        """
        detail = next(
            finding.detail
            for finding in self.findings(self.submission())
            if finding.name == "the path does not drift away from its velocity"
        )
        drift = float(detail.split()[3])

        self.assertLess(drift, 0.01 * CUMULATIVE_DRIFT_TOLERANCE_METRES)

    def test_a_run_that_stopped_at_the_horizon_is_not_an_accusation(self):
        """The other production ending. A run that reaches the horizon reports
        truncated and stops with the rocket still moving, so it has none of the
        repeated rows the trim looks for. Everything else here is a landing."""
        submission = self.submission()
        records = submission["balloon_world_data"]["trajectories"]
        flown = [
            index
            for index, record in enumerate(records)
            if np.isfinite(record["rocket_states"][:3]).all()
        ]
        del records[flown[0] + len(flown) // 3 :]

        failed = [
            f"{f.name}: {f.detail}" for f in self.findings(submission) if not f.ok
        ]

        self.assertEqual(failed, [])

    def test_a_second_control_history_also_has_room_to_spare(self):
        """The bound has to hold for any legal flight, not the passive one it
        was measured on. A commanded roll puts thrust transients and burnout
        under a different history."""
        parameters, given = load_scenario_parameters(0)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        agent = AttitudeRateControlAgent(
            given, rate_targets=[0.5, 0.0, 0.0], launch_time=1
        )
        observation, _ = env.reset(seed=parameters["scenario"]["random_seed"])
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _, terminated, truncated, _ = env.step(
                agent.get_action(observation)
            )
        submission = {
            "balloon_world_data": {"trajectories": copy.deepcopy(env.trajectories)}
        }

        findings = {f.name: f for f in self.findings(submission)}
        worst = float(findings["recorded velocity matches the path"].detail.split()[2])

        self.assertTrue(all(f.ok for f in findings.values()))
        self.assertLess(worst, 0.5 * VELOCITY_CONSISTENCY_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
