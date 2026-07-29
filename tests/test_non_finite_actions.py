"""What the environment does with an action it cannot use.

A policy that diverges emits NaN, and it is the most ordinary failure there is
while training one. Measured on the pin these tests were written against: a
single NaN in `tvc` was harmless, the run finished with the same score as a
clean one, while a NaN in `launch_inclination_heading` raised
`ValueError: All components of the initial state y0 must be finite` and ended
the evaluation. ActiveRocketPy #19 makes the actuator setters refuse a
non-finite command too, so without this the first kind becomes the second.

The rule here is that a field the environment cannot use is dropped, the
actuator keeps what it holds, and the episode carries on. A run is not lost to
one bad step.
"""

import logging
import signal
import unittest
from contextlib import contextmanager
from unittest import mock
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import (
        BalloonPoppingEnv,
        check_action,
    )
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

# A step takes milliseconds. This is a bound on a run that has stopped
# returning, not a performance assertion, so it is far above anything real.
STEP_TIME_LIMIT_SECONDS = 30.0


@contextmanager
def _returns_within(seconds):
    """Fail rather than hang if the block does not finish.

    A step count cannot bound a step that does not return, and this is exactly
    the shape that produced one: without the guard, a NaN on the first step
    after launch left the solver refusing to accept a step, and the test ran
    twenty two minutes before it was killed by hand.

    `SIGALRM` is delivered between bytecodes, and the solver's inner loop calls
    back into Python for the derivative, so it does arrive. Verified against the
    real hang: it fired at 3.00 s of a run that would not have ended. POSIX
    only, and skipped where the signal does not exist.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def ring(_signum, _frame):
        raise AssertionError(f"a step did not return within {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, ring)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


SCENARIO = 0
LAUNCH_STEP = 1
STEPS_AFTER_LAUNCH = 8


_CLEAN = {
    "launch": np.array(False),
    "launch_inclination_heading": np.array([80.0, 0.0]),
    "tvc": np.array([0.0, 0.0]),
    "roll": np.array([0.0]),
    "throttle": np.array([1.0]),
}


class TestWhichFieldsAreUnusable(unittest.TestCase):
    """The reader, tested on its own because everything below trusts it."""

    def test_a_clean_action_has_none(self):
        self.assertEqual(check_action(dict(_CLEAN)), [])

    def test_each_field_is_found_by_name(self):
        for field, shape in (
            ("launch", 1),
            ("launch_inclination_heading", 2),
            ("tvc", 2),
            ("roll", 1),
            ("throttle", 1),
        ):
            with self.subTest(field=field):
                action = dict(_CLEAN, **{field: np.full(shape, np.nan)})

                self.assertEqual(check_action(action), [field])

    def test_an_unreadable_launch_is_named_like_any_other_field(self):
        """It is read like the rest and it is reported like the rest. An agent
        whose launch never fires can otherwise see nothing at all: the field is
        silently no-launch and the four it does report are all fine."""
        for value in (np.nan, np.array([1, 1]), "nope", None):
            with self.subTest(value=repr(value)):
                self.assertEqual(check_action(dict(_CLEAN, launch=value)), ["launch"])

    def test_a_nested_array_of_the_right_size_is_flattened_not_refused(self):
        """`np.asarray` keeps the shape it was given, so the size test alone
        passed a (1, 2) that then raised at `float(tvc[0])`, and a (1, 2)
        attitude that reached `attitude[1]`. Two numbers in order are not
        ambiguous, so they are read rather than dropped."""
        for field, value in (
            ("tvc", np.array([[1.0, 2.0]])),
            ("tvc", np.array([[1.0], [2.0]])),
            ("launch_inclination_heading", np.array([[80.0, 90.0]])),
            ("roll", np.array([[3.0]])),
            ("launch", np.array([[1.0]])),
        ):
            with self.subTest(field=field, shape=value.shape):
                self.assertEqual(check_action(dict(_CLEAN, **{field: value})), [])

    def test_a_complex_field_is_not_usable(self):
        """`np.asarray(..., dtype=float)` discards the imaginary part with a
        warning rather than refusing, so a 1+9j gimbal command would arrive at
        the actuator as 1 and be indistinguishable from one the agent meant.

        The object cases are the ones a dtype test alone misses. `iscomplexobj`
        reads the array's dtype, and an object array does not carry its
        elements' types, so an array of `np.complex128` has dtype object,
        passes that test, and casts to 1 with only a warning all the same.
        """
        for label, value in (
            ("complex array", np.array([1 + 9j, 0 + 2j])),
            ("list of numpy complex", [np.complex128(1 + 9j), np.complex128(2j)]),
            (
                "object array of numpy complex",
                np.array([np.complex128(1 + 9j), np.complex128(2j)], dtype=object),
            ),
            (
                "object array of 0-d complex",
                np.array([np.array(1 + 9j), np.array(2j)], dtype=object),
            ),
            ("object array of floats", np.array([1.0, 2.0], dtype=object)),
        ):
            with self.subTest(value=label):
                self.assertEqual(check_action(dict(_CLEAN, tvc=value)), ["tvc"])

    def test_an_action_that_is_not_a_mapping_has_no_usable_field(self):
        """The lookup raises on these, and it happens after the step has already
        advanced the balloons."""
        for action in (None, 42, [1.0, 2.0], "launch"):
            with self.subTest(action=repr(action)):
                self.assertEqual(check_action(action), sorted(_CLEAN))

    def test_a_value_that_refuses_to_become_an_array_is_not_usable(self):
        """The competitors here are writing policies, and a torch tensor that
        still carries its graph raises `RuntimeError` out of `__array__`, not
        one of the conversion errors. Reproduced rather than imported, since
        torch is not a dependency of this package."""

        class StillNeedsDetaching:
            def __array__(self, dtype=None, copy=None):
                raise RuntimeError("Can't call numpy() on Tensor that requires grad")

        action = dict(_CLEAN, tvc=StillNeedsDetaching())

        self.assertEqual(check_action(action), ["tvc"])

    def test_infinity_counts_too(self):
        """`np.clip` would carry an infinity straight through, and the actuator
        would store it, so it is not only NaN that has to be caught."""
        action = dict(_CLEAN, tvc=np.array([np.inf, 0.0]))

        self.assertEqual(check_action(action), ["tvc"])

    def test_a_field_that_is_not_a_number_at_all(self):
        """`np.asarray(..., dtype=float)` raises on these rather than giving a
        NaN, so they need naming separately or they reach the actuator."""
        for value in ("nope", None, {"x": 1}):
            with self.subTest(value=repr(value)):
                self.assertIn("roll", check_action(dict(_CLEAN, roll=value)))

    def test_a_missing_field_is_not_usable(self):
        """Every shipped agent returns all five, and `step()` indexes them, so a
        partial action passing the helper and failing in the environment is the
        split this exists to remove."""
        self.assertEqual(
            check_action({"roll": np.array([0.0])}),
            ["launch", "launch_inclination_heading", "throttle", "tvc"],
        )

    def test_the_wrong_number_of_values_is_not_usable(self):
        """An empty array passes a finiteness test on its own, and `step()`
        indexes `tvc[1]`."""
        for field, value in (
            ("tvc", np.array([])),
            ("tvc", np.zeros(3)),
            ("tvc", 0.0),
            ("launch_inclination_heading", np.array([80.0])),
        ):
            with self.subTest(field=field, size=np.size(value)):
                self.assertIn(field, check_action(dict(_CLEAN, **{field: value})))


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestAnActionTheEnvironmentCannotUse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters, _given = load_scenario_parameters(SCENARIO)

    def _fresh(self):
        env = BalloonPoppingEnv(render_mode=None, parameters=self.parameters)
        env.reset(seed=self.parameters["scenario"]["random_seed"])
        return env

    @staticmethod
    def _action(env, launch=False):
        action = env.action_space.sample()
        action["launch"] = np.array(int(launch), dtype=action["launch"].dtype)
        action["launch_inclination_heading"] = np.array([80.0, 0.0], dtype=np.float64)
        action["throttle"] = np.ones_like(action["throttle"])
        action["tvc"] = np.zeros_like(action["tvc"])
        action["roll"] = np.zeros_like(action["roll"])
        return action

    def _flown(self, env, steps, corrupt=None):
        """Launch, then step, optionally corrupting one field throughout.

        Returns the environment and how the last step ended, because "the run
        carried on" is a claim about those flags rather than about the number of
        records: a terminal step appends one too.
        """
        terminated = truncated = False
        for index in range(LAUNCH_STEP + steps):
            action = self._action(env, launch=index == LAUNCH_STEP - 1)
            if corrupt and index >= LAUNCH_STEP:
                action[corrupt] = np.full_like(
                    np.asarray(action[corrupt], dtype=float), np.nan
                )
            with _returns_within(STEP_TIME_LIMIT_SECONDS):
                _observation, _reward, terminated, truncated, _info = env.step(action)
        return env, terminated, truncated

    def test_a_control_field_of_nans_does_not_end_the_run(self):
        """The case ActiveRocketPy #19 turns into a ValueError without this.

        Each step is given a wall clock bound as well. Without the guard this is
        the test that stops returning, and a run that never finishes reports
        nothing: it takes the CI job's whole budget to say so.
        """
        for field in ("tvc", "roll", "throttle"):
            with self.subTest(field=field):
                env, terminated, truncated = self._flown(
                    self._fresh(), STEPS_AFTER_LAUNCH, corrupt=field
                )

                self.assertTrue(env.rocket_launched)
                self.assertFalse(terminated, "the flight ended")
                self.assertFalse(truncated, "the episode ran out of horizon")
                self.assertEqual(
                    len(env.trajectories), LAUNCH_STEP + STEPS_AFTER_LAUNCH
                )

    def test_the_recorded_state_stays_finite_through_a_dropped_command(self):
        """Dropping the field has to keep the NaN out of the simulation, not
        only out of the score. A run whose recorded state is already NaN passes
        every check above.

        Every step after launch, not every step: the record for the step before
        the rocket exists is all NaN by design, and a clean run has the same one.
        """
        for field in ("tvc", "roll", "throttle"):
            with self.subTest(field=field):
                env, _terminated, _truncated = self._flown(
                    self._fresh(), STEPS_AFTER_LAUNCH, corrupt=field
                )
                states = np.asarray(
                    [record["rocket_states"] for record in env.trajectories],
                    dtype=float,
                )
                flown = states[LAUNCH_STEP:]

                self.assertEqual(len(flown), STEPS_AFTER_LAUNCH)
                self.assertTrue(np.isfinite(flown).all())

    def test_the_only_record_that_is_not_finite_is_the_one_before_launch(self):
        """The control for the test above: it would pass on a run that recorded
        nothing after launch, and it would look like a promise the environment
        does not make about the first record."""
        env, _terminated, _truncated = self._flown(self._fresh(), STEPS_AFTER_LAUNCH)
        states = np.asarray(
            [record["rocket_states"] for record in env.trajectories], dtype=float
        )
        finite = [index for index, row in enumerate(states) if np.isfinite(row).all()]

        self.assertEqual(finite, list(range(LAUNCH_STEP, len(states))))

    def test_the_actuator_keeps_the_value_it_had(self):
        """Dropping the command has to mean keeping the last one, not zeroing
        it, or a single NaN would silently cut the throttle."""
        env = self._fresh()
        for index in range(4):
            env.step(self._action(env, launch=index == LAUNCH_STEP - 1))
        held = float(env._rocket_flight.rocket.throttle_control.throttle)

        action = self._action(env)
        action["throttle"] = np.full_like(
            np.asarray(action["throttle"], dtype=float), np.nan
        )
        env.step(action)

        self.assertEqual(
            float(env._rocket_flight.rocket.throttle_control.throttle), held
        )

    def test_a_launch_attitude_of_nans_does_not_launch(self):
        """There is no previous attitude to fall back on, so the launch is
        refused rather than dropped. Measured without this: the flight is built
        from a NaN quaternion and raises out of the integrator."""
        env = self._fresh()
        action = self._action(env, launch=True)
        action["launch_inclination_heading"] = np.array([np.nan, 0.0])

        env.step(action)

        self.assertFalse(env.rocket_launched)

    def test_a_launch_that_fails_part_way_leaves_the_environment_unlaunched(self):
        """The flag is what the next step reads to decide which branch to take.

        Set before the flight is built, a failure in between leaves the
        environment marked as launched with no flight, and the next step
        dereferences `_rocket_flight.rocket`.
        """
        env = self._fresh()
        with mock.patch.object(
            BalloonPoppingEnv,
            "_BalloonPoppingEnv__init_rocket_simulation",
            side_effect=RuntimeError("the motor would not build"),
        ):
            with self.assertRaisesRegex(RuntimeError, "motor"):
                env.step(self._action(env, launch=True))

        self.assertFalse(env.rocket_launched)
        self.assertIsNone(env._rocket_flight)

    def test_the_agent_can_launch_on_a_later_step(self):
        """The half that stops the refusal above being a way to lose the run."""
        env = self._fresh()
        broken = self._action(env, launch=True)
        broken["launch_inclination_heading"] = np.array([np.nan, 0.0])
        env.step(broken)

        env.step(self._action(env, launch=True))

        self.assertTrue(env.rocket_launched)

    def test_the_dropped_field_is_named_in_the_log(self):
        """A competitor reading the output has to be able to tell that a command
        went nowhere, and which one."""
        env = self._fresh()
        for index in range(3):
            env.step(self._action(env, launch=index == LAUNCH_STEP - 1))
        action = self._action(env)
        action["tvc"] = np.full_like(np.asarray(action["tvc"], dtype=float), np.nan)

        with self.assertLogs(
            "BalloonPoppingGymEnv.envs.balloon_world", logging.WARNING
        ) as caught:
            env.step(action)

        self.assertIn("tvc", "".join(caught.output))

    def test_a_second_broken_field_is_still_named(self):
        """One counter for the whole action logged the 1st, 10th, 100th and
        1000th bad step. A field that broke after another had spent those
        milestones was never named once, which is the diagnostic this exists
        to give. Counted per field instead."""
        env = self._fresh()
        for index in range(3):
            action = self._action(env, launch=index == LAUNCH_STEP - 1)
            if index >= LAUNCH_STEP:
                action["throttle"] = np.full_like(
                    np.asarray(action["throttle"], dtype=float), np.nan
                )
            env.step(action)
        action = self._action(env)
        action["tvc"] = np.full_like(np.asarray(action["tvc"], dtype=float), np.nan)

        with self.assertLogs(
            "BalloonPoppingGymEnv.envs.balloon_world", logging.WARNING
        ) as caught:
            env.step(action)

        self.assertIn("tvc", "".join(caught.output))

    def test_a_nested_command_still_reaches_the_actuator(self):
        """Not only that it does not raise. A (1, 2) that is quietly dropped
        would pass a crash test while the agent loses control authority. Both
        angles are inside the 0.6 per step gimbal rate limit, so what arrives
        is what was asked for rather than what the actuator could reach."""
        env = self._fresh()
        for index in range(3):
            env.step(self._action(env, launch=index == LAUNCH_STEP - 1))

        action = self._action(env)
        action["tvc"] = np.array([[0.5, -0.25]])
        env.step(action)

        gimbal = env._rocket_flight.rocket.thrust_vector_control
        self.assertAlmostEqual(float(gimbal.gimbal_angle_x), 0.5)
        self.assertAlmostEqual(float(gimbal.gimbal_angle_y), -0.25)

    def test_a_nested_launch_attitude_still_launches(self):
        """The launch step has no previous value to fall back on, so refusing a
        shape that is not ambiguous costs the whole run rather than one step."""
        env = self._fresh()
        action = self._action(env, launch=True)
        action["launch_inclination_heading"] = np.array([[80.0, 0.0]])

        env.step(action)

        self.assertTrue(env.rocket_launched)

    def test_an_action_that_is_not_a_mapping_does_not_end_the_run(self):
        """`field not in action` raised on these, after the step had already
        advanced the balloons. An agent that returns None on one step keeps the
        actuators it had and flies on."""
        for value in (None, 42, "launch"):
            with self.subTest(action=repr(value)):
                env = self._fresh()
                for index in range(3):
                    env.step(self._action(env, launch=index == LAUNCH_STEP - 1))
                held = float(env._rocket_flight.rocket.throttle_control.throttle)

                env.step(value)

                self.assertTrue(env.rocket_launched)
                self.assertEqual(
                    float(env._rocket_flight.rocket.throttle_control.throttle), held
                )

    def test_a_clean_run_logs_nothing_and_flies_the_same(self):
        """The half that stops all of this being satisfied by dropping
        everything. The clean run commands `throttle` 1.0 every step and the
        corrupted one commands NaN, so the two land in the same place only
        because the dropped field holds; zeroing it would not fly there."""
        env = self._fresh()
        logger = logging.getLogger("BalloonPoppingGymEnv.envs.balloon_world")
        with self.assertNoLogs(logger, logging.WARNING):
            clean = self._where_it_got_to(env, STEPS_AFTER_LAUNCH)

        corrupted = self._where_it_got_to(
            self._fresh(), STEPS_AFTER_LAUNCH, corrupt="throttle"
        )

        self.assertTrue(env.rocket_launched)
        self.assertEqual(clean, corrupted)

    def _where_it_got_to(self, env, steps, corrupt=None):
        """Fly and return the last rocket position, so a run that quietly stops
        commanding is not equal to one that did not."""
        self._flown(env, steps, corrupt=corrupt)
        return env.trajectories[-1]["rocket_states"][:3]


if __name__ == "__main__":
    unittest.main()
