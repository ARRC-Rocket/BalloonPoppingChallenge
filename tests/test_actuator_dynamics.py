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
import math
import unittest
import warnings
from collections import namedtuple
from importlib.util import find_spec

import numpy as np

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO_NUMBER = 0
TIME_CONSTANT = 0.05
# Two time constants far enough apart that their responses cannot be confused.
# One value alone proves only that a lag of that size works: an implementation
# storing the requested constant correctly while computing every coefficient from
# a hard-coded 0.05 passes a whole suite written around 0.05.
FAST_TIME_CONSTANT = 0.02
SLOW_TIME_CONSTANT = 0.10
# Long enough for a first-order lag with this time constant to settle.
SETTLE_STEPS = 120

Actuator = namedtuple(
    "Actuator",
    "name time_constant_key rate_limit_key read read_public command initial",
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
        lambda rocket: rocket.thrust_vector_control.gimbal_angle_x,
        lambda action: _set_vector(action, 0),
        0.0,
    ),
    Actuator(
        "tvc_y",
        "gimbal_time_constant",
        "gimbal_rate_limit",
        lambda rocket: rocket.thrust_vector_control.y,
        lambda rocket: rocket.thrust_vector_control.gimbal_angle_y,
        lambda action: _set_vector(action, 1),
        0.0,
    ),
    Actuator(
        "roll",
        "roll_torque_time_constant",
        "torque_rate_limit",
        lambda rocket: rocket.roll_control,
        lambda rocket: rocket.roll_control.roll_torque,
        lambda action: _set_scalar(action, "roll"),
        0.0,
    ),
    Actuator(
        "throttle",
        "throttle_time_constant",
        "throttle_rate_limit",
        lambda rocket: rocket.throttle_control,
        lambda rocket: rocket.throttle_control.throttle,
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


def _launched_env(time_step=None, **time_constants):
    """Build scenario 0 with the given time constants and send the launch action.

    ``time_step`` can be overridden because scenario 0 ships 0.01 s, which makes
    ``1 / time_step`` exactly 100. Every expectation derived from it then agrees
    with a controller sampling rate hard-coded to 100, and the wiring this file
    exists to check becomes unobservable. Measured: replacing all three
    ``sampling_rate=1 / time_step`` expressions with the literal 100 passed the
    whole file.
    """
    parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
    parameters = copy.deepcopy(parameters)
    if time_step is not None:
        parameters["simulation"]["time_step"] = time_step
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


def _drive(env, action, actuator, value, steps, test_case=None):
    """Hold ``value`` on one actuator for ``steps`` steps, others left where they are.

    Records the Flight-facing property, not the underlying ``actuator_output``.
    Those are the same value through a passthrough today, but they are not the
    same contract: the environment writes ``roll_torque``, ``throttle`` and
    ``gimbal_angle_x/y``, and the equations of motion read those. A regression
    confined to one of those getters leaves ``actuator_output`` filtering
    correctly while the flight uses the wrong number. Measured: making
    ``roll_torque`` return zero whenever a time constant is set passed all 10
    tests here and all 61 in the suite.

    When a test case is supplied the two are compared every step, so the
    passthrough itself is pinned rather than assumed.
    """
    outputs = []
    for _ in range(steps):
        held = copy.deepcopy(action)
        held["launch"] = np.array(0, dtype=action["launch"].dtype)
        _hold_current_commands(env, held)
        actuator.command(held)(value)
        env.step(held)
        rocket = env._rocket_flight.rocket
        public = float(actuator.read_public(rocket))
        if test_case is not None:
            test_case.assertAlmostEqual(
                public,
                float(actuator.read(rocket).actuator_output),
                places=12,
                msg=f"{actuator.name}: the flight-facing value left actuator_output",
            )
        outputs.append(public)
    return outputs


def _per_step_limit(parameters, actuator):
    control = parameters["rocket"]["control"]
    return control[actuator.rate_limit_key] * parameters["simulation"]["time_step"]


def _command_within_limit(parameters, actuator, sign=1.0):
    """A step well inside the rate limit, so the lag is what shows.

    Throttle starts at the top of its range, so it only has room downwards; the
    others are centred at zero and are exercised both ways.
    """
    half_step = 0.5 * _per_step_limit(parameters, actuator)
    if actuator.name == "throttle":
        sign = -1.0
    return actuator.initial + sign * half_step


def _directions(actuator):
    """Which way this actuator can be driven from its initial output."""
    return (-1.0,) if actuator.name == "throttle" else (1.0, -1.0)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestActuatorWiring(unittest.TestCase):
    """The scenario's timing and time constants have to reach the actuators."""

    def test_demand_rate_matches_the_scenario_timestep(self):
        """The environment steps every ``time_step``, so the actuators must too.

        Driven at a step scenario 0 does not ship. Its 0.01 s makes
        ``1 / time_step`` exactly 100, so every expectation derived from it also
        matches a sampling rate hard-coded to 100, and this test would say
        nothing. 0.025 s gives 40 Hz, which no plausible literal produces.
        """
        for time_step, expected in ((0.025, 40.0), (0.005, 200.0)):
            with self.subTest(time_step=time_step):
                env, _action, parameters = _launched_env(time_step=time_step)

                self.assertEqual(parameters["simulation"]["time_step"], time_step)
                rocket = env._rocket_flight.rocket
                for actuator in ACTUATORS:
                    with self.subTest(actuator=actuator.name):
                        self.assertEqual(actuator.read(rocket).demand_rate, expected)

    def test_the_response_follows_the_scenario_timestep_too(self):
        """Not just the stored rate: the filter has to use it.

        Storing the right demand_rate while updating at some other cadence would
        satisfy the assertion above. The discrete response depends on the step
        through ``alpha = dt / (tau + dt)``, so driving at a non-default step
        and predicting from it covers the behaviour rather than the attribute.
        """
        time_step = 0.025
        steps = 2
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                env, action, parameters = _launched_env(
                    time_step=time_step, **{actuator.time_constant_key: TIME_CONSTANT}
                )
                command = _command_within_limit(parameters, actuator)
                outputs = _drive(env, action, actuator, command, steps, test_case=self)

                alpha = time_step / (TIME_CONSTANT + time_step)
                fraction = (outputs[-1] - actuator.initial) / (
                    command - actuator.initial
                )
                self.assertAlmostEqual(fraction, 1 - (1 - alpha) ** steps, places=6)

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
        """Reading the YAML is enough; no need to build an environment."""
        for scenario_number in (0, 1):
            parameters, _ = load_scenario_parameters(scenario_number)
            control = parameters["rocket"]["control"]
            for actuator in ACTUATORS:
                with self.subTest(scenario=scenario_number, actuator=actuator.name):
                    self.assertIsNone(control[actuator.time_constant_key])


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestActuatorDynamics(unittest.TestCase):
    """Every actuator has to show the lag its time constant models."""

    def test_a_held_command_lags_then_settles_on_it(self):
        for actuator in ACTUATORS:
            for sign in _directions(actuator):
                with self.subTest(actuator=actuator.name, sign=sign):
                    env, action, parameters = _launched_env(
                        **{actuator.time_constant_key: TIME_CONSTANT}
                    )
                    command = _command_within_limit(parameters, actuator, sign)
                    outputs = _drive(
                        env, action, actuator, command, SETTLE_STEPS, test_case=self
                    )
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

                    # One time constant in, a first-order lag has covered about
                    # 1 - 1/e of the span. Checking that rather than a particular
                    # alpha keeps this independent of which discretisation upstream
                    # uses, while still rejecting a wrong coefficient: backward
                    # Euler gives 0.598 here and the exact exponential 0.632, but a
                    # hard-coded alpha of 0.25 reaches 0.763 and a filter that lags
                    # once and then snaps reaches 1.0.
                    steps_per_tau = round(
                        TIME_CONSTANT / parameters["simulation"]["time_step"]
                    )
                    at_one_tau = (outputs[steps_per_tau - 1] - actuator.initial) / span
                    self.assertGreater(
                        at_one_tau, 0.55, "settles too slowly for this tau"
                    )
                    self.assertLess(at_one_tau, 0.70, "settles too fast for this tau")

                    self.assertAlmostEqual(outputs[-1], command, delta=abs(span) * 1e-3)

    def test_without_a_time_constant_the_command_lands_at_once(self):
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                env, action, parameters = _launched_env()
                command = _command_within_limit(parameters, actuator)
                for output in _drive(env, action, actuator, command, 2, test_case=self):
                    self.assertAlmostEqual(output, command, places=6)

    def test_a_larger_time_constant_gives_a_slower_response(self):
        """The scenario's value has to be the one driving the filter.

        Every other dynamics test here uses a single time constant, so a
        coefficient computed from a hard-coded 0.05 satisfies all of them while
        ignoring what the scenario asked for. Two constants compared at the same
        elapsed time is what separates those.

        The expectation is the exact discrete response, ``1 - (1 - alpha)**n``
        with ``alpha = dt / (tau + dt)``, derived from the scenario's own time
        step and requested constant. Not the continuous ``1 - exp(-t/tau)``: at
        the few steps needed to keep the two responses well apart, the two differ
        by more than any tolerance worth calling tight.
        """
        steps = 2
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name):
                responses = {}
                for time_constant in (FAST_TIME_CONSTANT, SLOW_TIME_CONSTANT):
                    env, action, parameters = _launched_env(
                        **{actuator.time_constant_key: time_constant}
                    )
                    command = _command_within_limit(parameters, actuator)
                    outputs = _drive(
                        env, action, actuator, command, steps, test_case=self
                    )

                    fraction = (outputs[-1] - actuator.initial) / (
                        command - actuator.initial
                    )
                    responses[time_constant] = fraction

                    alpha = parameters["simulation"]["time_step"] / (
                        time_constant + parameters["simulation"]["time_step"]
                    )
                    self.assertAlmostEqual(fraction, 1 - (1 - alpha) ** steps, places=6)

                fast = responses[FAST_TIME_CONSTANT]
                slow = responses[SLOW_TIME_CONSTANT]
                self.assertGreater(
                    fast,
                    slow,
                    "the configured time constant is not driving the response",
                )
                # Separated, not merely ordered: a filter ignoring the request
                # would put both at the same fraction.
                self.assertGreater(fast / slow, 2.0)

    def test_the_rate_limit_hands_back_to_the_lag(self):
        """The whole transition, sample by sample, not the first two.

        The documented composition is: filter the command, then rate-limit the
        *filtered* change. That produces full-limit steps only while the filtered
        demand outruns the limit, then a first-order decay. Asserting that a
        handover merely happens is not enough: a limiter that rate-limits the raw
        command instead takes full steps for twice as long and then decays the
        same way, so it hands back too, just later.

        Compared against a reference written straight from that description. It
        shares no code with the actuator, so agreeing with it pins the order the
        two stages are applied in and the point where the second takes over.
        """
        for actuator in ACTUATORS:
            for sign in (1.0, -1.0):
                with self.subTest(actuator=actuator.name, sign=sign):
                    self._check_handover(actuator, sign)

    def _check_handover(self, actuator, sign):
        env, action, parameters = _launched_env(
            **{actuator.time_constant_key: TIME_CONSTANT}
        )
        step = parameters["simulation"]["time_step"]
        limit = _per_step_limit(parameters, actuator)
        alpha = step / (TIME_CONSTANT + step)
        start = actuator.initial
        if actuator.name == "throttle":
            # It starts at the top of its range, so it needs room to go up.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*exceeds rate limit.*")
                reached = _drive(
                    env,
                    action,
                    actuator,
                    actuator.initial - 20 * limit,
                    20,
                    test_case=self,
                )
            start = reached[-1]
        command = start + sign * 10 * limit

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*exceeds rate limit.*")
            outputs = _drive(
                env, action, actuator, command, SETTLE_STEPS, test_case=self
            )

        expected = []
        output = start
        for _ in range(SETTLE_STEPS):
            filtered = alpha * command + (1 - alpha) * output
            change = filtered - output
            if abs(change) > limit:
                filtered = output + math.copysign(limit, change)
            output = filtered
            expected.append(output)

        np.testing.assert_allclose(outputs, expected, rtol=1e-9, atol=1e-9)

        # And the shape that reference describes, stated outright so a
        # reader does not have to run the loop in their head.
        sizes = [
            abs(later - earlier) for earlier, later in zip([start] + outputs, outputs)
        ]
        released = next(i for i, size in enumerate(sizes) if size < limit - 1e-9)
        self.assertGreater(released, 0, "the limit never bound at all")
        # Starting 10 limits away and stepping by exactly one limit while
        # the limiter binds, the distance before step i is (10 - i) limits
        # and the filtered demand is alpha times that. It drops *below*
        # the limit at the first i with 10 - i < 1/alpha, and the step
        # where the two are exactly equal is still a full one.
        self.assertEqual(released, math.floor(10 - 1 / alpha) + 1)
        for earlier, later in zip(sizes[released:], sizes[released + 1 :]):
            self.assertLessEqual(later, earlier + 1e-12)
        self.assertAlmostEqual(outputs[-1], command, delta=abs(command) * 1e-3)
        self.assertLessEqual(
            max(sign * (value - command) for value in outputs),
            1e-9,
            "the response overshot",
        )

    def test_the_rate_limit_clips_both_directions(self):
        """Clipping is a sign-independent bound, and only one sign was covered.

        The large-command test drives TVC and roll positive and throttle
        negative, so an implementation clipping only ``change > max_change``
        keeps every gimbal and roll case green.
        """
        for actuator in ACTUATORS:
            for sign in (1.0, -1.0):
                with self.subTest(actuator=actuator.name, sign=sign):
                    env, action, parameters = _launched_env()
                    limit = _per_step_limit(parameters, actuator)
                    start = actuator.initial

                    if actuator.name == "throttle":
                        # It starts at the top of its range, so there is no room
                        # upwards. Walk it into the interior first, one legal
                        # step at a time, then command from there.
                        interior_steps = 20
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", message=r".*exceeds rate limit.*"
                            )
                            reached = _drive(
                                env,
                                action,
                                actuator,
                                actuator.initial - interior_steps * limit,
                                interior_steps,
                                test_case=self,
                            )
                        start = reached[-1]
                        self.assertGreater(start, 0.0)
                        self.assertLess(start, 1.0)

                    with self.assertWarnsRegex(UserWarning, "exceeds rate limit"):
                        outputs = _drive(
                            env, action, actuator, start + sign * 10 * limit, 2
                        )

                    self.assertAlmostEqual(abs(outputs[0] - start), limit, places=6)
                    self.assertAlmostEqual(abs(outputs[1] - start), 2 * limit, places=6)
                    # The direction has to follow the command, not a fixed sign.
                    self.assertEqual(np.sign(outputs[0] - start), sign)

    def test_the_rate_limit_bounds_a_large_command(self):
        """A command far past the limit is bound by it, which hides the lag.

        Worth pinning: it is why the tests above stay inside the limit.
        """
        # Both scenarios ship with the dynamics off, so the limit has to hold
        # there too: a regression that only rate-limits when a time constant is
        # configured would otherwise pass every test in this file.
        for time_constant in (None, TIME_CONSTANT):
            for actuator in ACTUATORS:
                with self.subTest(actuator=actuator.name, tau=time_constant):
                    env, action, parameters = _launched_env(
                        **{actuator.time_constant_key: time_constant}
                    )
                    limit = _per_step_limit(parameters, actuator)
                    sign = -1.0 if actuator.name == "throttle" else 1.0
                    # Clipping is expected to say so, so pin that as well.
                    with self.assertWarnsRegex(UserWarning, "exceeds rate limit"):
                        outputs = _drive(
                            env,
                            action,
                            actuator,
                            actuator.initial + sign * 10 * limit,
                            2,
                            test_case=self,
                        )
                    self.assertAlmostEqual(
                        abs(outputs[0] - actuator.initial), limit, places=6
                    )
                    self.assertAlmostEqual(
                        abs(outputs[1] - actuator.initial), 2 * limit, places=6
                    )


# Which state the flight should move, and which it should leave alone. y_sol is
# pos(3), vel(3), quaternion(4), body rates(3), so index 10 is the first body
# rate. gimbal_angle_x drives that one and gimbal_angle_y the next; measured,
# with the cross-axis staying at 1e-9 while the driven one reaches 0.86.
FLIGHT_STATE = {"tvc_x": 10, "tvc_y": 11, "roll": 12, "throttle": 5}
CROSS_AXIS = {"tvc_x": 11, "tvc_y": 10}


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheDynamicsReachTheFlight(unittest.TestCase):
    """The last link: the filtered value moves the rocket, not just the object.

    Everything else ends at the actuator, or at the property the flight reads.
    Neither says the equations of motion use it. A flight that ignored an
    actuator, or used it only when its dynamics were off, satisfied all of it,
    and both shipped scenarios leave the time constants null so the golden
    masters would not notice either.

    Compared differentially rather than pinned: the same command with the lag
    off and on, and the driven state at the same step. No trajectory is frozen,
    so this does not become another golden master.

    Signed, not absolute. Reversing a moment's sign only on the path with
    dynamics enabled leaves both magnitudes ordered the same way, so an
    absolute comparison passes it.
    """

    def _state_after(self, actuator, sign, time_constant, steps=30):
        env, action, parameters = _launched_env(
            **{actuator.time_constant_key: time_constant}
        )
        limit = _per_step_limit(parameters, actuator)
        command = actuator.initial + sign * 12 * limit

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*exceeds rate limit.*")
            _drive(env, action, actuator, command, steps, test_case=self)

        state = np.asarray(env._rocket_states, dtype=float)
        driven = float(state[FLIGHT_STATE[actuator.name]])
        cross = (
            float(state[CROSS_AXIS[actuator.name]])
            if actuator.name in CROSS_AXIS
            else None
        )
        rocket = env._rocket_flight.rocket
        return driven, cross, float(actuator.read_public(rocket))

    def test_every_actuator_moves_its_own_state_and_the_lag_delays_it(self):
        for actuator in ACTUATORS:
            # Throttle starts at the top of its range, so down is the only way.
            sign = -1.0 if actuator.name == "throttle" else 1.0
            # Throttle down means less thrust, so the vertical velocity that
            # results is still positive but smaller; the other three move their
            # state the way they are commanded.
            expected_sign = 1.0
            with self.subTest(actuator=actuator.name):
                # Commanding nothing, as the other end of the interval the
                # lagged response has to fall inside.
                held, _held_cross, _held_value = self._state_after(actuator, 0.0, None)
                immediate, immediate_cross, immediate_value = self._state_after(
                    actuator, sign, None
                )
                lagged, _lagged_cross, lagged_value = self._state_after(
                    actuator, sign, TIME_CONSTANT
                )

                # The premise, stated so a state failure is not ambiguous about
                # whether the command or the physics is at fault.
                self.assertNotAlmostEqual(
                    immediate_value,
                    lagged_value,
                    msg="the lag did not change the command, so this proves nothing",
                )
                # Something happened at all.
                self.assertGreater(abs(immediate), 1e-3)
                # The direction the command asked for, absolutely rather than
                # relative to the other run. Reversing a moment's sign on both
                # paths keeps every comparison below intact.
                self.assertEqual(
                    np.sign(immediate),
                    expected_sign,
                    "the state moved the opposite way from the command",
                )
                self.assertEqual(np.sign(immediate), np.sign(lagged))

                # A lag delays a response; it neither deletes one nor skips it.
                # The lagged state has to sit strictly between commanding
                # nothing and commanding immediately, and both ends are needed.
                #
                # Without the floor, applying the actuator only when its
                # dynamics are off passes everything else: measured, that left
                # the lagged body rate at 1e-24. Without the ceiling, ignoring
                # the actuator's value only when they are on passes too: for
                # throttle that means full thrust, which moves the state the
                # same way the lag does, only further.
                self.assertGreater(abs(immediate - lagged), 0.01 * abs(immediate))
                low, high = sorted((held, immediate))
                self.assertGreater(
                    lagged, low, "the lagged response overshot the commanded one"
                )
                self.assertLess(
                    lagged, high, "the lagged response never left the held state"
                )

                if immediate_cross is not None:
                    # The axis it must not move.
                    self.assertLess(abs(immediate_cross), 1e-6 * abs(immediate))


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestActuatorStateAcrossEpisodes(unittest.TestCase):
    """A new episode starts from clean actuators, not the previous one's state.

    ActiveRocketPy unit tests cover an actuator's own reset. What is checked
    here is the environment path: reset() drops the flight, and the next launch
    builds a fresh rocket, so filter state must not survive the episode.
    """

    def test_relaunching_starts_the_filters_from_their_initial_output(self):
        constants = {a.time_constant_key: TIME_CONSTANT for a in ACTUATORS}
        env, action, parameters = _launched_env(**constants)

        # Drive every actuator away from where it started.
        for actuator in ACTUATORS:
            _drive(
                env, action, actuator, _command_within_limit(parameters, actuator), 20
            )
        moved = env._rocket_flight.rocket
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name, phase="driven"):
                self.assertNotAlmostEqual(
                    float(actuator.read(moved).actuator_output), actuator.initial
                )

        env.reset(seed=parameters["scenario"]["random_seed"])
        self.assertIsNone(env._rocket_flight, "reset should drop the flight")

        relaunch = env.action_space.sample()
        relaunch["launch"] = np.array(1, dtype=relaunch["launch"].dtype)
        relaunch["launch_inclination_heading"] = np.array(
            [90.0, 0.0], dtype=relaunch["launch_inclination_heading"].dtype
        )
        _hold_current_commands(env, relaunch)
        env.step(relaunch)

        rocket = env._rocket_flight.rocket
        for actuator in ACTUATORS:
            with self.subTest(actuator=actuator.name, phase="relaunched"):
                self.assertAlmostEqual(
                    float(actuator.read(rocket).actuator_output),
                    actuator.initial,
                    places=9,
                )
                self.assertEqual(
                    actuator.read(rocket).actuator_time_constant, TIME_CONSTANT
                )


if __name__ == "__main__":
    unittest.main()
