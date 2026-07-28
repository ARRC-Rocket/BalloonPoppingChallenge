"""What the agent gets back must not be the environment's own state.

``evaluate_scenario`` hands the observation straight to ``agent.get_action()``,
and the agent is a competitor's code. ``step()`` then computes the reward from
``_balloon_status`` after that call returns, and ``pack_for_submission`` records
the resulting count as the score. So an array handed out by reference is not an
API tidiness question, it is the scoring path.

Measured before the fix, on scenario 0: an agent that never sent a launch action
and only wrote 2 into the status array it was given scored 10 out of 10.

What this is not
----------------
It is not a boundary against hostile agent code, and the name of the first class
here used to say it was. ``_load_agent_class`` runs the submitted module with
``exec_module`` in the evaluator's own interpreter, and ``get_action`` is called
from ``evaluate_scenario`` while ``env`` is a local variable there, so an agent
reaches the environment through the caller frame whatever these arrays are.
Measured on the fixed code, so this is a limit rather than a regression:

    frame = inspect.currentframe().f_back
    frame.f_locals["env"]._balloon_status[:, 0] = 2

scored 10 out of 10 with the rocket never launched, and monkey patching at
import time is available before that. What is asserted below is the narrower
and still worthwhile thing: the data the environment hands out is a copy, so
holding onto it does not hold onto the environment.

Two separate limits, which are worth not running together. Running the agent in
another process, or a suitably constrained container, would close this
particular path: the evaluator's frame and its ``env`` object would no longer be
in the agent's interpreter to reach for. What it would not do is make a result
produced on the competitor's own machine tamper resistant, since the machine
owner still controls the parent process, the runtime, the filesystem and the
file that gets uploaded.

For that second one the answer is checking the submission afterwards. #97
proposes a script for it, which is not on this branch. It is worth being exact
about the size of that too: it regenerates the balloons from the shipped
scenario and asks whether the claimed pops were anywhere the rocket says it
went, so it catches this no-launch forged score and several related edits. It
does not verify the rocket trajectory itself, which stays the competitor's
claim.
"""

import copy
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_0_PARAMS = (
    REPO_ROOT
    / "BalloonPoppingGymEnv"
    / "envs"
    / "scenario_parameters"
    / "scenario_0_parameters.yaml"
)

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import yaml

    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestWhatTheEnvironmentHandsOutIsDetached(unittest.TestCase):
    def setUp(self):
        self.parameters = yaml.safe_load(
            SCENARIO_0_PARAMS.read_text(encoding="utf-8-sig")
        )
        self.env = BalloonPoppingEnv(render_mode=None, parameters=self.parameters)
        self.observation, self.info = self.env.reset(
            seed=self.parameters["scenario"]["random_seed"]
        )

    def _idle_action(self):
        action = self.env.action_space.sample()
        action["launch"] = np.array(0, dtype=action["launch"].dtype)
        for key in ("tvc", "roll"):
            action[key] = np.zeros_like(action[key])
        # Held at its initial output rather than zeroed, so this does not also
        # drive the throttle rate limiter.
        action["throttle"] = np.ones_like(action["throttle"])
        return action

    def test_nothing_handed_out_shares_memory_with_the_environment(self):
        """The property, rather than one consequence of losing it.

        Checked on both what reset returns and what step returns, since the two
        go through the same accessors but a future change could make only one of
        them safe.
        """
        pairs = [
            (
                "reset balloon_status",
                self.observation["balloon_status"],
                self.env._balloon_status,
            ),
            (
                "reset balloon_states",
                self.observation["balloon_states"],
                self.env._balloon_states,
            ),
            (
                "reset rocket_sensors",
                self.observation["rocket_sensors"],
                self.env._rocket_sensors,
            ),
            (
                "reset rocket_states",
                self.info["rocket_states"],
                self.env._rocket_states,
            ),
        ]
        observation, _reward, _terminated, _truncated, info = self.env.step(
            self._idle_action()
        )
        pairs += [
            (
                "step balloon_status",
                observation["balloon_status"],
                self.env._balloon_status,
            ),
            (
                "step balloon_states",
                observation["balloon_states"],
                self.env._balloon_states,
            ),
            (
                "step rocket_sensors",
                observation["rocket_sensors"],
                self.env._rocket_sensors,
            ),
            ("step rocket_states", info["rocket_states"], self.env._rocket_states),
        ]

        for label, handed_out, internal in pairs:
            with self.subTest(field=label):
                self.assertFalse(
                    np.shares_memory(handed_out, internal),
                    f"{label} is a writable alias of the environment's state",
                )

    def test_writing_the_status_back_does_not_score(self):
        """The exploit, driven the way an agent would.

        Three lines, no launch, and before the fix it returned the whole
        scenario. The rocket never leaves the pad, so nothing about this run
        could legitimately pop anything.
        """
        self.observation["balloon_status"][:] = 2

        observation, reward, _terminated, _truncated, info = self.env.step(
            self._idle_action()
        )

        self.assertFalse(self.env.rocket_launched, "this run should never launch")
        self.assertEqual(reward, 0)
        self.assertEqual(info["popped_count"], 0)
        self.assertTrue(
            (np.asarray(observation["balloon_status"]) != 2).all(),
            "a balloon is recorded as popped",
        )

    def test_writing_the_balloon_positions_back_does_not_move_them(self):
        """``_balloon_states`` is a slice of the precomputed flight array.

        Handed out by reference, writing to it rewrote the trajectory the pop
        sweep compares the rocket against, which is the same exploit by another
        route: put the balloons on the rocket rather than the score on the
        balloons.
        """
        before = self.env._balloon_flights.copy()

        self.observation["balloon_states"][:] = 12345.0

        np.testing.assert_array_equal(self.env._balloon_flights, before)
        self.assertTrue(
            (np.asarray(self.env._balloon_states) != 12345.0).all(),
            "the environment's balloon states were rewritten",
        )

    def test_writing_the_sensors_and_rocket_state_back_changes_nothing(self):
        observation, _reward, _terminated, _truncated, info = self.env.step(
            self._idle_action()
        )
        sensors_before = self.env._rocket_sensors.copy()
        states_before = self.env._rocket_states.copy()

        observation["rocket_sensors"][:] = 999.0
        info["rocket_states"][:] = 999.0

        np.testing.assert_array_equal(self.env._rocket_sensors, sensors_before)
        np.testing.assert_array_equal(self.env._rocket_states, states_before)

    def test_rewriting_the_observation_every_step_scores_nothing(self):
        """End to end, because a single step is not where a score is decided.

        A caller that rewrites everything it is handed on every call, and never
        launches. It should finish with the same score as one that does nothing
        at all, which is none.

        Named for what it drives. It writes to the arrays here rather than from
        inside an agent the evaluator loaded, so it establishes that the handed
        out arrays are detached and nothing about what agent code can reach.
        """
        action = self._idle_action()
        observation = self.observation
        steps = 0
        while True:
            observation["balloon_status"][:] = 2
            observation["balloon_states"][:] = 0.0
            observation["rocket_sensors"][:] = 0.0
            observation, _reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            if terminated or truncated:
                break
            self.assertLess(steps, self.env.num_timesteps + 5, "episode did not end")

        self.assertEqual(info["popped_count"], 0)
        self.assertFalse(self.env.rocket_launched)


# Sharing one of these is what lets a write travel. A tuple is descended into
# rather than reported, since holding the same tuple is harmless and holding a
# list inside one is not.
_MUTABLE_CONTAINERS = (list, dict, set, bytearray)


def _reachable_mutables(root):
    """Every mutable container reachable from ``root``, by identity and path.

    Iterative and with a ``seen`` set, so a structure that refers to itself
    terminates instead of recursing until the stack runs out. ``deepcopy`` keeps
    a memo for exactly that reason, and a helper written to check ``deepcopy``
    should survive the same input.
    """
    found = {}
    seen = set()
    stack = [(root, "<root>")]
    while stack:
        value, path = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, _MUTABLE_CONTAINERS):
            found[id(value)] = path
        if isinstance(value, dict):
            stack.extend((child, f"{path}.{key}") for key, child in value.items())
        elif isinstance(value, (list, tuple)):
            stack.extend(
                (child, f"{path}[{index}]") for index, child in enumerate(value)
            )
    return found


def _shared_objects(given, scenario):
    """Paths in ``given`` holding a mutable object that ``scenario`` also holds.

    Both trees are walked in full and compared by identity, rather than walking
    them together and comparing matching positions. Walking together looked
    equivalent and was not: it only ever compared a key against the same key and
    an index against the same index, so it saw nothing when the same object sat
    under a different key, at a shifted index, or past the end of the shorter
    side of a ``zip``. Measured, all three came back clean:

        shared = []
        _shared_objects({"a": shared}, {"b": shared})          -> []
        _shared_objects({"a": [None, shared]}, {"a": [shared]}) -> []
        _shared_objects({"a": [1, shared]}, {"a": [1]})        -> []

    Which is the whole property this is for. The point is not that a particular
    key was copied, it is that nothing the agent is handed is an object the
    environment still reads.

    A tuple is descended into rather than reported, since holding the same tuple
    is harmless and holding a list inside one is not.
    """
    theirs = _reachable_mutables(scenario)
    return sorted(
        path
        for identity, path in _reachable_mutables(given).items()
        if identity in theirs
    )


class TestTheWalkerFindsSharingItIsPointedAt(unittest.TestCase):
    """The check above is only worth what this one says it is.

    No simulation stack needed: these are hand built trees, which is the point.
    A test whose only input is a tree that already passes cannot tell a working
    walker from one that returns an empty list.
    """

    def test_a_shared_list_is_reported(self):
        shared = [1, 2]

        self.assertEqual(_shared_objects({"a": shared}, {"a": shared}), ["<root>.a"])

    def test_a_shared_dict_of_plain_values_is_reported(self):
        """The one that went missing entirely, because it looked like a branch."""
        shared = {"x": 1}

        self.assertEqual(_shared_objects({"a": shared}, {"a": shared}), ["<root>.a"])

    def test_sharing_inside_a_copied_list_is_reported(self):
        """The shallow copy regression this exists for.

        ``original.copy()`` gives the outer list a new identity and leaves every
        element the same object. Stopping at the container calls that clean.
        """
        inner = [1, 2]
        original = [inner, {"x": 3}]

        found = _shared_objects({"a": list(original)}, {"a": original})

        self.assertEqual(found, ["<root>.a[0]", "<root>.a[1]"])

    def test_sharing_under_a_different_key_is_reported(self):
        """Walking the two trees together never compared these at all.

        The property is that nothing handed out is an object the environment
        still reads, not that a particular key was copied.
        """
        shared = [1, 2]

        self.assertEqual(_shared_objects({"a": shared}, {"b": shared}), ["<root>.a"])

    def test_sharing_at_a_shifted_index_is_reported(self):
        shared = [1, 2]

        found = _shared_objects({"a": [None, shared]}, {"a": [shared]})

        self.assertEqual(found, ["<root>.a[1]"])

    def test_sharing_past_the_shorter_sequence_is_reported(self):
        """zip() stopped at the shorter side and never looked at the rest.

        Both sides have to actually hold the object for this to be sharing.
        ``{"a": [1, shared]}`` against ``{"a": [1]}`` is not a case: only one
        side holds it, and an empty result there is the right answer.
        """
        shared = [1, 2]

        found = _shared_objects({"a": [0, 0, shared]}, {"a": [shared]})

        self.assertEqual(found, ["<root>.a[2]"])

    def test_a_structure_that_refers_to_itself_terminates(self):
        """It recursed until the stack ran out.

        No scenario YAML does this today, and a helper written to check
        ``deepcopy`` should survive what ``deepcopy`` itself handles.
        """
        original = []
        original.append(original)

        self.assertEqual(_shared_objects(copy.deepcopy(original), original), [])
        self.assertEqual(_shared_objects(original, original), ["<root>"])

    def test_a_deep_copy_is_reported_clean(self):
        """Or reporting everything would pass the tests above."""
        original = {"a": [[1, 2], {"x": 3}], "b": {"c": [4]}}

        self.assertEqual(_shared_objects(copy.deepcopy(original), original), [])

    def test_equal_but_distinct_objects_are_not_sharing(self):
        self.assertEqual(_shared_objects({"a": [1, 2]}, {"a": [1, 2]}), [])


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheGivenParametersAreAlsoDetached(unittest.TestCase):
    """The other thing the agent is handed, and it reaches further.

    The observation is what the agent sees every step; ``given_parameters`` is
    what its constructor gets. The extraction builds new dicts, so replacing a
    value there never reached the environment, but any list value was the same
    object the environment reads.

    Measured: four were shared, and two of them are read when the flight is
    built on the launch action, which is after the agent has already run. So an
    agent could widen its own throttle range or change the rocket's moment of
    inertia from its ``__init__``.
    """

    def setUp(self):
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        self.scenario, self.given = load_scenario_parameters(0)

    def test_nothing_in_them_is_the_same_object(self):
        """The property, checked over the whole tree rather than the two known
        offenders, so a new whitelisted key cannot reintroduce this quietly."""
        self.assertEqual(
            _shared_objects(self.given, self.scenario),
            [],
            "given_parameters shares objects with the scenario",
        )

    def test_writing_to_them_does_not_reach_the_environment(self):
        """Driven the way an agent would, on the two that reached the physics."""
        env = BalloonPoppingEnv(render_mode=None, parameters=self.scenario)
        throttle_before = list(env.rocket_parameters["control"]["throttle_range"])
        inertia_before = list(env.rocket_parameters["rocket_body"]["inertia"])

        self.given["rocket"]["control"]["throttle_range"][1] = 99.0
        self.given["rocket"]["rocket_body"]["inertia"][0] = 1e-6

        self.assertEqual(
            list(env.rocket_parameters["control"]["throttle_range"]), throttle_before
        )
        self.assertEqual(
            list(env.rocket_parameters["rocket_body"]["inertia"]), inertia_before
        )

    def test_the_values_are_still_the_ones_the_scenario_holds(self):
        """Or the fix could be "hand the agent something else entirely"."""
        self.assertEqual(
            self.given["rocket"]["control"]["throttle_range"],
            self.scenario["rocket"]["control"]["throttle_range"],
        )
        self.assertEqual(
            self.given["rocket"]["rocket_body"]["inertia"],
            self.scenario["rocket"]["rocket_body"]["inertia"],
        )


if __name__ == "__main__":
    unittest.main()
