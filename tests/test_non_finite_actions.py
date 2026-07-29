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
import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import (
        BalloonPoppingEnv,
        check_action,
    )
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO = 0
LAUNCH_STEP = 1
STEPS_AFTER_LAUNCH = 8


_CLEAN = {
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
            ("launch_inclination_heading", 2),
            ("tvc", 2),
            ("roll", 1),
            ("throttle", 1),
        ):
            with self.subTest(field=field):
                action = dict(_CLEAN, **{field: np.full(shape, np.nan)})

                self.assertEqual(check_action(action), [field])

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
        """Every shipped agent returns all four, and `step()` indexes them, so a
        partial action passing the helper and failing in the environment is the
        split this exists to remove."""
        self.assertEqual(
            check_action({"roll": np.array([0.0])}),
            ["launch_inclination_heading", "throttle", "tvc"],
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
        """Launch, then step, optionally corrupting one field throughout."""
        for index in range(LAUNCH_STEP + steps):
            action = self._action(env, launch=index == LAUNCH_STEP - 1)
            if corrupt and index >= LAUNCH_STEP:
                action[corrupt] = np.full_like(
                    np.asarray(action[corrupt], dtype=float), np.nan
                )
            env.step(action)
        return env

    def test_a_control_field_of_nans_does_not_end_the_run(self):
        """The case ActiveRocketPy #19 turns into a ValueError without this."""
        for field in ("tvc", "roll", "throttle"):
            with self.subTest(field=field):
                env = self._flown(self._fresh(), STEPS_AFTER_LAUNCH, corrupt=field)

                self.assertTrue(env.rocket_launched)
                self.assertEqual(
                    len(env.trajectories), LAUNCH_STEP + STEPS_AFTER_LAUNCH
                )

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

    def test_a_clean_run_logs_nothing_and_scores_the_same(self):
        """The half that stops all of this being satisfied by dropping
        everything."""
        env = self._fresh()
        logger = logging.getLogger("BalloonPoppingGymEnv.envs.balloon_world")
        with self.assertNoLogs(logger, logging.WARNING):
            for index in range(LAUNCH_STEP + STEPS_AFTER_LAUNCH):
                env.step(self._action(env, launch=index == LAUNCH_STEP - 1))

        self.assertTrue(env.rocket_launched)


if __name__ == "__main__":
    unittest.main()
