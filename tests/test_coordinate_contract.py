"""The coordinate datum the competition API exposes.

X and Y are measured from the launch point, but Z is altitude above sea level:
the rocket starts at ``z = elevation`` and a balloon 10 m above the pad reads
``elevation + 10``. Upstream RocketPy documents launch-site-relative positions
instead, so this is a fork-level divergence that a future submodule update could
silently reverse. Competitors aim with these numbers, so it is worth pinning.

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


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestCoordinateDatum(unittest.TestCase):
    def setUp(self):
        parameters, _ = load_scenario_parameters(SCENARIO_NUMBER)
        # The fixture this test needs, stated here rather than borrowed from the
        # scenario. A distinctive non-zero elevation, because at elevation 0 sea
        # level and the pad are the same point and every ASL-versus-pad-relative
        # assertion below stops distinguishing anything. Zero GNSS noise and a
        # zero receiver offset, because the comparisons are exact. Tuning the
        # scenario for realism would otherwise either delete the coverage
        # silently or fail this test for a reason that is not a frame change.
        self.parameters = copy.deepcopy(parameters)
        self.parameters["environment"]["elevation"] = 123.0
        sensors = self.parameters["rocket"]["sensors"]
        sensors["gnss_position"] = 0.0
        sensors["gnss_position_accuracy"] = 0.0
        sensors["gnss_altitude_accuracy"] = 0.0
        sensors["gnss_velocity_accuracy"] = 0.0
        self.elevation = float(self.parameters["environment"]["elevation"])
        self.env = BalloonPoppingEnv(render_mode=None, parameters=self.parameters)
        self.observation, _ = self.env.reset(
            seed=self.parameters["scenario"]["random_seed"]
        )

    def test_balloon_heights_are_above_sea_level(self):
        """Scenario 0 places balloon i at elevation + 10 + 40i."""
        count = int(self.parameters["balloon"]["num"])
        expected_above_pad = 10.0 + 40.0 * np.arange(count)
        np.testing.assert_allclose(
            self.env._balloon_flights[:, 2, 0] - self.elevation, expected_above_pad
        )
        # The same numbers reach the agent, not a pad-relative version of them.
        np.testing.assert_allclose(
            self.observation["balloon_states"][:, 2],
            expected_above_pad + self.elevation,
        )

    def test_the_internal_state_starts_at_the_launch_point(self):
        launch = self._launch()

        np.testing.assert_allclose(
            np.asarray(launch[1:3], dtype=float), [0.0, 0.0], rtol=0, atol=1e-9
        )

    def test_the_rocket_starts_at_the_site_elevation(self):
        launch = self._launch()

        self.assertAlmostEqual(float(launch[3]), self.elevation, places=9)

    def test_the_observed_balloon_horizontal_positions_are_launch_relative(self):
        """Scenario 0 stacks its balloons directly over the pad."""
        np.testing.assert_allclose(
            self.observation["balloon_states"][:, :2], 0.0, rtol=0, atol=1e-9
        )

    def test_the_observed_horizontal_position_is_launch_relative(self):
        """The public GNSS, which is what an agent actually navigates on.

        Asserting this on ``initial_solution`` says nothing about it: that is
        the integrator's own state, and the observation is produced separately.
        A global offset added to the reported X and Y, or reporting latitude and
        longitude there, left every other test in this file green.

        Scenario 0's GNSS accuracies are all zero, so this is exact rather than
        within a noise budget.
        """
        self._launch()
        observation, _reward, _term, _trunc, _info = self.env.step(self._idle())
        gnss_xy = np.asarray(observation["rocket_sensors"][6:8], dtype=float)

        # One step of a vertical launch: still over the pad, and small either way
        # compared with the site's latitude and longitude, or with any offset
        # large enough to matter to an agent.
        np.testing.assert_allclose(gnss_xy, [0.0, 0.0], rtol=0, atol=1.0)

    def test_the_reported_altitude_is_anchored_to_the_site_elevation(self):
        """An absolute anchor, because the two public channels agreeing is not one.

        The observation and the info are produced at two different points, so
        comparing them only with each other says they agree, not that either is
        right. A common offset survives it: 100 m added to both the GNSS
        altitude and the reported state passed every other assertion here.

        Nor is the flight's own solution a third opinion. ``_rocket_states`` is
        ``y_sol[:]`` on an ndarray, which shares memory with it, so comparing
        against it compares a value with itself. Measured with
        ``np.shares_memory``.

        The anchor is the elevation this fixture sets. One step after launch the
        rocket has climbed 2.5 mm, so the reported altitude has to be the site
        elevation to within a few centimetres, and no common offset can hide in
        that.
        """
        self._launch()
        observation, _reward, _term, _trunc, info = self.env.step(self._idle())
        gnss = np.asarray(observation["rocket_sensors"][6:9], dtype=float)
        state = np.asarray(info["rocket_states"], dtype=float)[:3]

        self.assertAlmostEqual(float(gnss[2]), self.elevation, delta=0.1)
        self.assertAlmostEqual(float(state[2]), self.elevation, delta=0.1)
        # And the two channels agree, which the anchor above does not imply.
        np.testing.assert_allclose(gnss, state, rtol=0, atol=1e-6)

    def test_the_observed_velocity_is_in_the_same_frame(self):
        """A frame change would show up here as a rotated or offset velocity."""
        self._launch()
        observation, _reward, _term, _trunc, info = self.env.step(self._idle())
        gnss_velocity = np.asarray(observation["rocket_sensors"][9:12], dtype=float)
        state_velocity = np.asarray(info["rocket_states"], dtype=float)[3:6]

        np.testing.assert_allclose(gnss_velocity, state_velocity, rtol=0, atol=1e-6)
        # Anchored the same way: one step after launch it has barely moved.
        np.testing.assert_allclose(gnss_velocity, [0.0, 0.0, 0.0], rtol=0, atol=1.0)

    def test_the_two_horizontal_axes_are_not_interchangeable(self):
        """X is east and Y is north, and a vertical launch cannot tell them apart.

        Every other case here launches straight up, so the reported X and Y both
        sit at zero and swapping them changes nothing. Fly at an angle until the
        two are far apart, then compare each against its own true component.

        60 degrees on a heading of 30 was picked by measuring: it separates the
        two by 24.8 m before the flight ends. 45 degrees comes down after 0.11 m,
        and a heading of 45 keeps X and Y equal by construction, which would have
        looked like a working test and proved nothing.
        """
        self._launch(inclination=60.0, heading=30.0)
        observation, info = None, None
        for _ in range(600):
            observation, _reward, terminated, truncated, info = self.env.step(
                self._idle()
            )
            state = np.asarray(info["rocket_states"], dtype=float)
            if abs(state[0] - state[1]) > 5.0:
                break
            if terminated or truncated:
                self.fail("the flight ended before the two axes separated")

        state = np.asarray(info["rocket_states"], dtype=float)[:3]
        self.assertGreater(
            abs(state[0] - state[1]),
            5.0,
            "the axes never separated, so a swap would still be invisible",
        )

        gnss = np.asarray(observation["rocket_sensors"][6:9], dtype=float)
        np.testing.assert_allclose(gnss, state, rtol=0, atol=1e-6)

    def test_the_gnss_accuracies_really_are_zero_here(self):
        """The precondition for the exact comparisons above.

        If scenario 0 ever gains sensor noise, those become flaky rather than
        wrong, and this says which it is.
        """
        sensors = self.parameters["rocket"]["sensors"]

        for key in (
            "gnss_position_accuracy",
            "gnss_altitude_accuracy",
            "gnss_velocity_accuracy",
        ):
            with self.subTest(setting=key):
                self.assertEqual(float(sensors[key]), 0.0)

    def _idle(self):
        """A step that disturbs nothing, which is not the same as commanding zero.

        Throttle starts at 1.0 and its rate limit is 2.0 per second over a 0.01 s
        step, so asking for zero demands twenty times what one step allows: the
        actuator warns, and the thrust this test is not interested in changes
        underneath it. Hold throttle where it is and zero the two that already
        rest at zero.
        """
        action = self.env.action_space.sample()
        action["launch"] = np.array(0, dtype=action["launch"].dtype)
        for key in ("tvc", "roll"):
            action[key] = np.zeros_like(action[key])
        action["throttle"] = np.full_like(action["throttle"], self._current_throttle())
        return action

    def _current_throttle(self):
        flight = self.env._rocket_flight
        if flight is None:
            return 1.0
        return float(flight.rocket.throttle_control.actuator_output)

    def _launch(self, inclination=90.0, heading=0.0):
        action = self._idle()
        action["launch"] = np.array(1, dtype=action["launch"].dtype)
        action["launch_inclination_heading"] = np.array(
            [inclination, heading], dtype=action["launch_inclination_heading"].dtype
        )
        self.env.step(action)
        return self.env.initial_solution
