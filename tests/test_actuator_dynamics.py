"""Actuator dynamics wiring between a scenario and the flight.

Both shipped scenarios leave every actuator time constant at ``null``, so the
golden masters only ever cover the pass-through path. These tests set non-null
time constants and check that they reach the actuators the flight drives, and
that the first-order lag they model is actually visible in the output.

The actuator classes themselves are unit tested in ActiveRocketPy; what is
covered here is this package's side of the contract.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import copy
import unittest
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0
TIME_CONSTANT = 0.05


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
    for key in ("tvc", "throttle", "roll"):
        action[key] = np.zeros_like(action[key])
    env.step(action)
    return env, action, parameters["rocket"]["control"]


def _gimbal_outputs(env, action, command, steps):
    """Hold a gimbal command for ``steps`` steps and return the actuator output."""
    outputs = []
    for _ in range(steps):
        held = copy.deepcopy(action)
        held["launch"] = np.array(0, dtype=action["launch"].dtype)
        held["tvc"] = np.array([command, 0.0], dtype=action["tvc"].dtype)
        env.step(held)
        outputs.append(
            float(env._rocket_flight.rocket.thrust_vector_control.x.actuator_output)
        )
    return outputs


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestActuatorDynamicsWiring(unittest.TestCase):
    """Scenario time constants have to reach the actuators the flight drives."""

    def test_non_null_time_constants_reach_every_actuator(self):
        env, _action, _control = _launched_env(
            gimbal_time_constant=0.05,
            roll_torque_time_constant=0.08,
            throttle_time_constant=0.12,
        )
        rocket = env._rocket_flight.rocket

        # The dual-axis TVC holds one single-axis actuator per gimbal axis.
        self.assertEqual(rocket.thrust_vector_control.x.actuator_time_constant, 0.05)
        self.assertEqual(rocket.thrust_vector_control.y.actuator_time_constant, 0.05)
        self.assertEqual(rocket.roll_control.actuator_time_constant, 0.08)
        self.assertEqual(rocket.throttle_control.actuator_time_constant, 0.12)

    def test_null_time_constants_leave_the_filter_off(self):
        """The shipped scenarios use null, which has to stay pass-through."""
        env, _action, control = _launched_env()
        self.assertIsNone(control["gimbal_time_constant"], "scenario 0 changed")
        rocket = env._rocket_flight.rocket

        for actuator in (
            rocket.thrust_vector_control.x,
            rocket.thrust_vector_control.y,
            rocket.roll_control,
            rocket.throttle_control,
        ):
            self.assertIsNone(actuator.actuator_time_constant)
            self.assertEqual(actuator._alpha, 1.0)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestActuatorDynamicsBehaviour(unittest.TestCase):
    """A non-null time constant has to be visible in the gimbal output."""

    # Small enough that the rate limit never binds, so the lag is what shows.
    SMALL_COMMAND = 0.3

    def test_output_lags_and_converges_towards_the_command(self):
        env, action, control = _launched_env(gimbal_time_constant=TIME_CONSTANT)
        demand_period = 1.0 / control_demand_rate(env)
        expected_alpha = demand_period / (TIME_CONSTANT + demand_period)

        outputs = _gimbal_outputs(env, action, self.SMALL_COMMAND, steps=4)

        # First sample is the filtered fraction of the command, not the command.
        self.assertAlmostEqual(
            outputs[0], expected_alpha * self.SMALL_COMMAND, places=6
        )
        self.assertLess(outputs[0], self.SMALL_COMMAND)
        # Holding the command walks the output towards it without overshooting.
        for earlier, later in zip(outputs, outputs[1:]):
            self.assertGreater(later, earlier)
            self.assertLess(later, self.SMALL_COMMAND)

    def test_without_a_time_constant_the_same_command_is_immediate(self):
        env, action, _control = _launched_env()
        outputs = _gimbal_outputs(env, action, self.SMALL_COMMAND, steps=2)
        for output in outputs:
            self.assertAlmostEqual(output, self.SMALL_COMMAND, places=6)

    def test_the_rate_limit_still_bounds_a_large_command(self):
        """With a command far above the rate limit the limit is what binds.

        Worth pinning: it means a large step hides the time constant entirely,
        which is why the tests above use a command below the per-step limit.
        """
        env, action, control = _launched_env(gimbal_time_constant=TIME_CONSTANT)
        per_step_limit = control["gimbal_rate_limit"] / control_demand_rate(env)
        large_command = float(control["max_gimbal_angle"])

        outputs = _gimbal_outputs(env, action, large_command, steps=2)

        self.assertAlmostEqual(outputs[0], per_step_limit, places=6)
        self.assertAlmostEqual(outputs[1], 2 * per_step_limit, places=6)


def control_demand_rate(env):
    """The actuator demand rate the environment configures, in Hz."""
    return env._rocket_flight.rocket.thrust_vector_control.x.demand_rate


if __name__ == "__main__":
    unittest.main()
