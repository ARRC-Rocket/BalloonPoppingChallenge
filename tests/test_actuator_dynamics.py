"""Actuator dynamics wiring between a scenario and the flight.

Both shipped scenarios leave every actuator time constant at ``null``, so the
golden masters only ever cover the pass-through path. These tests set non-null
time constants and check that they reach the actuators the flight drives, and
that the first-order lag they model is visible in every actuator's output.

Expected values are derived from the scenario's ``time_step`` rather than read
back from the actuator under test. Reading them back would make a wiring mistake
self-validating: configuring the controllers at half the sampling rate changes
the actuator output and the expectation together, so the test stays green while
the physical response is wrong.

The actuator classes themselves are unit tested in ActiveRocketPy; what is
covered here is this package's side of the contract.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import copy
import unittest
from collections import namedtuple
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0
TIME_CONSTANT = 0.05
# Long enough for a first-order lag with this time constant to settle.
SETTLE_STEPS = 120

Actuator = namedtuple(
    "Actuator", "name time_constant_key rate_limit_key read command initial"
)


def _set_vector(action, index):
    def setter(value):
        action["tvc"] = np.array(
            [value if i == index else action["tvc"][i] for i in range(2)],
            dtype=action["tvc"].dtype,
        )

    return setter


def _set_scalar(action, key):
    def setter(value):
        action[key] = np.asarray(value, dtype=action[key].dtype).reshape(
            np.shape(action[key])
        )

    return setter


ACTUATORS = (
    Actuator(
        "tvc_x",
        "gimbal_time_constant",
        "gimbal_rate_limit",
        lambda rocket: rocket.thrust_vector_control.x,
        lambda action: _set_vector(action, 0),
        0.0,
    ),
    Actuator(
        "tvc_y",
        "gimbal_time_constant",
        "gimbal_rate_limit",
        lambda rocket: rocket.thrust_vector_control.y,
        lambda action: _set_vector(action, 1),
        0.0,
    ),
    Actuator(
        "roll",
        "roll_torque_time_constant",
        "torque_rate_limit",
        lambda rocket: rocket.roll_control,
        lambda action: _set_scalar(action, "roll"),
        0.0,
    ),
    Actuator(
        "throttle",
        "throttle_time_constant",
        "throttle_rate_limit",
        lambda rocket: rocket.throttle_control,
        lambda action: _set_scalar(action, "throttle"),
        1.0,
    ),
)


def _hold_current_commands(env, action):
    """Command every actuator its own current output.

    Commanding a neutral zero instead would drive throttle down from its initial
    1.0 on every step, so a gimbal test would quietly be a throttle rate-limit
    test as well, warnings included.
    """
    rocket = env._rocket_flight.rocket if env._rocket_flight is not None else None
    for actuator in ACTUATORS:
        current = (
            actuator.initial
            if rocket is None
            else float(actuator.read(rocket).actuator_output)
        )
        actuator.command(action)(current)


def _launched_env(**time_constants):
    """Build scenario 0 with the given time constants and send the launch action."""
    parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
    parameters = copy.deepcopy(parameters)
    parameters["rocket"]["control"].update(time_constants)

    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    env.reset(seed=parameters["scenario"]["random_seed"])

    action = env.action_space.sample()
    action["launch"] = np.array(1, dtype=action["launch"].dtype)
    action["launch_inclination_heading"] = np.array(
        [90.0, 0.0], dtype=action["launch_inclination_heading"].dtype
    )
    _hold_current_commands(env, action)
    env.step(action)
    return env, action, parameters


def _drive(env, action, actuator, value, steps):
    """Hold ``value`` on one actuator for ``steps`` steps, others left where they are."""
    outputs = []
    for _ in range(steps):
        held = copy.deepcopy(action)
        held["launch"] = np.array(0, dtype=action["launch"].dtype)
        _hold_current_commands(env, held)
        actuator.command(held)(value)
        env.step(held)
        outputs.append(float(actuator.read(env._rocket_flight.rocket).actuator_output))
    return outputs


def _per_step_limit(parameters, actuator):
    control = parameters["rocket"]["control"]
    return control[actuator.rate_limit_key] * parameters["simulation"]["time_step"]


def _command_within_limit(parameters, actuator):
    """A step well inside the rate limit, so the lag is what shows."""
    half_step = 0.5 * _per_step_limit(parameters, actuator)
    return actuator.initial + (-half_step if actuator.name == "throttle" else half_step)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestActuatorWiring(unittest.TestCase):
    """The scenario's timing and time constants have to reach the actuators."""

    def test_demand_rate_matches_the_scenario_timestep(self):
        """The environment steps every ``time_step``, so the actuators must too.

        Without this, halving the configured sampling rate would leave every
        behaviour test below green while the real response is wrong.
        """
        env, _action, parameters = _launched_env()
        expected = 1.0 / parameters["simulation"]["time_step"]
        rocket = env._rocket_flight.rocket
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                self.assertEqual(actuator.read(rocket).demand_rate, expected)

    def test_non_null_time_constants_reach_every_actuator(self):
        env, _action, _parameters = _launched_env(
            gimbal_time_constant=0.05,
            roll_torque_time_constant=0.08,
            throttle_time_constant=0.12,
        )
        rocket = env._rocket_flight.rocket
        expected = {"tvc_x": 0.05, "tvc_y": 0.05, "roll": 0.08, "throttle": 0.12}
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                self.assertEqual(
                    actuator.read(rocket).actuator_time_constant,
                    expected[actuator.name],
                )

    def test_shipped_scenarios_leave_the_dynamics_off(self):
        _env, _action, parameters = _launched_env()
        control = parameters["rocket"]["control"]
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                self.assertIsNone(control[actuator.time_constant_key])


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestActuatorDynamics(unittest.TestCase):
    """Every actuator has to show the lag its time constant models."""

    def test_a_held_command_lags_then_settles_on_it(self):
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                env, action, parameters = _launched_env(
                    **{actuator.time_constant_key: TIME_CONSTANT}
                )
                command = _command_within_limit(parameters, actuator)
                outputs = _drive(env, action, actuator, command, SETTLE_STEPS)
                span = command - actuator.initial

                first_fraction = (outputs[0] - actuator.initial) / span
                self.assertGreater(first_fraction, 0.0, "no response at all")
                self.assertLess(first_fraction, 0.5, "no lag: it arrived at once")

                for earlier, later in zip(outputs, outputs[1:]):
                    self.assertGreaterEqual(
                        (later - earlier) / span, 0.0, "moved away from the command"
                    )
                    self.assertLessEqual(
                        (later - actuator.initial) / span, 1.0 + 1e-9, "overshoot"
                    )

                self.assertAlmostEqual(outputs[-1], command, delta=abs(span) * 1e-3)

    def test_without_a_time_constant_the_command_lands_at_once(self):
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                env, action, parameters = _launched_env()
                command = _command_within_limit(parameters, actuator)
                for output in _drive(env, action, actuator, command, 2):
                    self.assertAlmostEqual(output, command, places=6)

    def test_the_rate_limit_bounds_a_large_command(self):
        """A command far past the limit is bound by it, which hides the lag.

        Worth pinning: it is why the tests above stay inside the limit.
        """
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                env, action, parameters = _launched_env(
                    **{actuator.time_constant_key: TIME_CONSTANT}
                )
                limit = _per_step_limit(parameters, actuator)
                sign = -1.0 if actuator.name == "throttle" else 1.0
                outputs = _drive(
                    env, action, actuator, actuator.initial + sign * 10 * limit, 2
                )
                self.assertAlmostEqual(
                    abs(outputs[0] - actuator.initial), limit, places=6
                )
                self.assertAlmostEqual(
                    abs(outputs[1] - actuator.initial), 2 * limit, places=6
                )


if __name__ == "__main__":
    unittest.main()
