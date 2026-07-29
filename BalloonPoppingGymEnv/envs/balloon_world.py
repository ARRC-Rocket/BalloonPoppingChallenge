import copy
import logging
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm
from gymnasium import spaces
from rocketpy import (
    Environment,
    Flight,
    LinearGenericSurface,
    MonteCarlo,
    Rocket,
    SolidMotor,
    StochasticEnvironment,
    StochasticFlight,
    StochasticRocket,
)
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gnss_receiver import GnssReceiver
from rocketpy.sensors.gyroscope import Gyroscope
from rocketpy.tools import euler313_to_quaternions

logger = logging.getLogger(__name__)

# How many numbers each action field holds. The environment indexes `tvc` and
# `launch_inclination_heading` by position, so a wrong count is not usable
# either, and an empty array passes a finiteness test on its own.
ACTION_FIELD_SIZES = {
    "launch": 1,
    "launch_inclination_heading": 2,
    "tvc": 2,
    "roll": 1,
    "throttle": 1,
}


# Which repetition of an unusable field gets a log line. A broken policy stays
# broken, and one line per step for the rest of the horizon is its own problem.
_UNUSABLE_ACTION_LOG_STEPS = frozenset({1, 10, 100, 1000})


def _wants_launch(commands):
    """Whether a readable `launch` asks for one.

    Any nonzero number launches, the way Python reads any other number. NaN is
    not readable, so it never reaches here and never launches.
    """
    return "launch" in commands and float(commands["launch"][0]) != 0.0


def check_action(action):
    """The action fields the environment cannot use, by name.

    A policy that diverges emits NaN, and it is the most ordinary failure there
    is while training one. An empty list means every field is usable.

    Public so an agent can ask the same question before returning an action:

        if check_action(action):
            action = last_good_action
    """
    return sorted(_usable_action_fields(action)[1])


def _usable_action_fields(action):
    """The fields as flat arrays of the right size, and the names of the rest.

    Validating a converted copy and then handing the original to the actuators
    would check something other than what is used: a scalar `tvc` converts to a
    finite array and then fails at `action["tvc"][0]`. So the converted values
    are what the caller gets.
    """
    # Nothing to index. Raising here instead would end the run over exactly the
    # kind of bad action the rest of this exists to survive.
    if not isinstance(action, Mapping):
        return {}, set(ACTION_FIELD_SIZES)
    usable = {}
    unusable = set()
    for field, size in ACTION_FIELD_SIZES.items():
        if field not in action:
            unusable.add(field)
            continue
        try:
            given = np.asarray(action[field])
            # Before the cast, which discards the imaginary part with a warning
            # rather than refusing it: 1+9j would gimbal to 1.
            if np.iscomplexobj(given):
                unusable.add(field)
                continue
            # Flat, because the size alone let a (1, 2) through to raise at
            # `float(tvc[0])` and a (1, 2) attitude to raise at `attitude[1]`.
            values = np.asarray(given, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            unusable.add(field)
            continue
        if values.size != size or not np.all(np.isfinite(values)):
            unusable.add(field)
            continue
        usable[field] = values
    return usable, unusable


def _remove_monte_carlo_workspace(directory):
    """Delete a Monte Carlo scratch directory, saying so if it will not go.

    ``ignore_errors=True`` would turn a failed removal back into the silent
    accumulation this exists to stop, and letting the error out would turn a
    finished reset into a failure over scratch files nobody wanted. Neither, so
    it is reported instead.
    """
    try:
        shutil.rmtree(directory)
    except OSError as error:
        logger.warning(
            "Could not remove the Monte Carlo workspace at %s: %s", directory, error
        )


def _monte_carlo_workspace_has_files(directory):
    """Whether the workspace holds anything worth keeping, without raising.

    This runs inside an ``except`` block. An error raised here would be chained
    onto the failure being handled rather than replacing it, so the original is
    still readable, but it would stop the path being logged and stop the empty
    directory being removed. So it answers "keep it" when it cannot tell.
    """
    try:
        return bool(os.listdir(directory))
    except OSError as error:
        logger.warning(
            "Could not inspect the Monte Carlo workspace at %s: %s", directory, error
        )
        return True


# The result keys the balloon flights are built from. Counted together rather
# than trusting one of them to speak for the rest.
_MONTE_CARLO_RESULT_KEYS = ("x", "y", "z", "vx", "vy", "vz", "lat0", "lon0")


def _check_monte_carlo_returned_every_balloon(results, requested):
    """Refuse a short result set instead of building a world out of it.

    ``simulate()`` does not always raise when it stops early. The pinned
    RocketPy catches ``KeyboardInterrupt``, appends to its error file, prints
    "Keyboard interrupt received. Files saved." and returns normally, so a
    Ctrl-C part way through a hundred balloons comes back as fewer trajectories
    rather than as a failure. Only ``Exception`` is re-raised there.

    Exactly one returned trajectory is the case worth naming. The release shift
    indexes with ``arange(num_balloons)`` against one release step per requested
    balloon, so a ``(1, 6, T)`` array broadcasts to a full ``(100, 6, T)`` one in
    which every balloon flies the same path, time shifted. Nothing downstream can
    tell, and the episode scores normally against a world that was never
    simulated. Every other short count reaches the same expression and raises an
    IndexError about indexing array shapes, which is loud but says nothing about
    what actually went wrong.
    """
    counts = {
        key: len(results[key]) if key in results else None
        for key in _MONTE_CARLO_RESULT_KEYS
    }
    if any(count != requested for count in counts.values()):
        raise RuntimeError(
            f"the balloon Monte Carlo was asked for {requested} trajectories "
            f"and returned {counts}, so the balloon flights cannot be built. A "
            "keyboard interrupt during the run is the usual cause: RocketPy "
            "catches it, saves its files and returns what it has."
        )


def _seed_sequence_to_int(seed_sequence):
    """Encode 128 generated state bits of ``seed_sequence`` as a plain int.

    Sensors keep whatever seed they were built with and hand it back from
    ``to_dict()``, which the submission packer runs through ``RocketPyEncoder``.
    A ``SeedSequence`` object has no JSON form, so passing one straight to a
    sensor makes packing a submission fail. An int has one and ``default_rng``
    accepts it.

    Not a change of representation. ``default_rng(child)`` mixes in the child's
    ``spawn_key``; ``default_rng(int)`` builds a fresh root sequence from these
    bits, so the generator draws a different stream from the one the child
    itself would have produced. That is deliberate: both shipped scenarios run
    at ``noise_density: 0.0``, so no baseline depends on the old stream, and the
    three derived values are pinned by a test so a later move is a decision
    rather than a side effect. Preserving the exact stream needs
    ``SeedSequence`` serialization upstream (RocketPy-Team/RocketPy#1087).

    The words are combined by value rather than through ``tobytes``, so the
    result does not depend on the machine's byte order.
    """
    words = seed_sequence.generate_state(4, dtype=np.uint32)
    return sum(int(word) << (32 * position) for position, word in enumerate(words))


class BalloonPoppingEnv(gym.Env):
    metadata = {"render_modes": ["vpython", "matplotlib"]}

    def __init__(self, render_mode, parameters):

        self.scenario_parameters = parameters["scenario"]
        self.environment_parameters = parameters["environment"]
        self.simulation_parameters = parameters["simulation"]
        self.balloon_parameters = parameters["balloon"]
        self.rocket_parameters = parameters["rocket"]

        # ActiveRocketPy flight classes for rocket flight and balloon flights
        self._rocket_flight = None
        self._balloon_flights = None

        # initial solution: [time, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        self.initial_solution = None
        # [balloon_num, status] status: 0-ground; 1-released; 2-popped
        self._balloon_status = np.zeros((self.balloon_parameters["num"], 1), dtype=int)
        # [balloon_num, (x, y, z, vx, vy, vz)]
        self._balloon_states = np.array(np.zeros((self.balloon_parameters["num"], 6)))
        # (gyroX, gyroY, gyroZ, accX, accY, accZ, posX, posY, posZ, velX, velY, velZ)
        self._rocket_sensors = np.full(12, np.nan)
        # Unusable steps per field, for rate limiting. Per field, because one
        # counter let a field that broke second go unnamed for the whole run.
        self._unusable_action_counts = Counter()
        # (posX, posY, posZ, velX, velY, velZ, e0, e1, e2, e3, w1, w2, w3)
        self._rocket_states = np.full(13, np.nan)
        # Where the next pop sweep starts. Tracked separately from
        # ``_rocket_states`` because the flight only produces a state on the step
        # after the launch action, while the sweep has to start at the pad.
        self._sweep_origin = None
        # save trajectories for logging
        self.trajectories = None

        # attributes for step()
        self.rocket_launched = False
        self.current_step = 0
        self.num_timesteps = 0
        self._popped_count = 0
        self._balloon_release_at_step = None

        self._rocketpy_env = None

        # Observations include balloon and rocket states
        self.observation_space = spaces.Dict(
            {
                "simulation_time": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(),
                    dtype=np.float64,
                ),
                "balloon_status": spaces.MultiDiscrete(
                    3 * np.ones((self.balloon_parameters["num"], 1), dtype=int)
                ),
                "balloon_states": spaces.Box(
                    low=-np.inf * np.ones((self.balloon_parameters["num"], 6)),
                    high=np.inf * np.ones((self.balloon_parameters["num"], 6)),
                    dtype=np.float64,
                ),
                "rocket_sensors": spaces.Box(
                    low=-np.inf * np.ones(12),
                    high=np.inf * np.ones(12),
                    dtype=np.float64,
                ),
            }
        )

        # tvc, roll, and throttling actions
        self.action_space = spaces.Dict(
            {
                "launch": spaces.Box(low=0, high=1, shape=(), dtype=bool),
                "launch_inclination_heading": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([90, 360]),
                    shape=(2,),
                    dtype=np.float64,
                ),
                "tvc": spaces.Box(
                    low=-self.rocket_parameters["control"]["max_gimbal_angle"]
                    * np.ones(2),
                    high=self.rocket_parameters["control"]["max_gimbal_angle"]
                    * np.ones(2),
                    dtype=np.float64,
                ),
                "throttle": spaces.Box(
                    low=self.rocket_parameters["control"]["throttle_range"][0],
                    high=self.rocket_parameters["control"]["throttle_range"][1],
                    shape=(),
                    dtype=np.float64,
                ),
                "roll": spaces.Box(
                    low=-self.rocket_parameters["control"]["max_roll_torque"],
                    high=self.rocket_parameters["control"]["max_roll_torque"],
                    shape=(),
                    dtype=np.float64,
                ),
            }
        )

        # Graphics-related attributes
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_canvas = None
        self.render_balloons = None
        self.render_rocket = None

    def _get_obs(self):
        """The agent's view of the world, detached from the world.

        Copies, not the arrays themselves. What is handed back goes straight to
        ``agent.get_action()``, and a numpy array handed out by reference is a
        writable alias: whatever the agent does to it happens to the
        environment's own state.

        That mattered here more than it usually does, because ``step()``
        computes the reward from ``_balloon_status`` after the agent has
        returned, and ``pack_for_submission`` records the resulting count as the
        score. An agent that never launched and only wrote 2 into the status it
        was handed scored every balloon in the scenario. ``_balloon_states`` is
        a slice of ``_balloon_flights``, so writing to that rewrote the balloon
        trajectory the pop sweep compares against as well.

        The cost is one copy of a few kilobytes per step, which is not
        measurable against a flight step.
        """
        sim_time = self.current_step * self.simulation_parameters["time_step"]
        return {
            "simulation_time": sim_time,
            "balloon_status": self._balloon_status.copy(),
            "balloon_states": self._balloon_states.copy(),
            "rocket_sensors": self._rocket_sensors.copy(),
        }

    def _get_info(self):
        """Same rule. ``_rocket_states`` aliases the flight's own solution
        vector, so handing it out by reference lets a caller write into the
        integrator's state between steps.
        """
        return {
            "rocket_states": self._rocket_states.copy(),
            "popped_count": int(self._popped_count),
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create ActiveRocketPy environment for balloon flights and rocket simulation
        self.__create_environment()
        self._rocket_flight = None
        self._balloon_flights = None
        self.initial_solution = None

        # Generate balloon release sequences for all balloons
        self.__reset_balloon_release_sequence()

        # Scenario 0: hello world with static balloons -- no Monte Carlo needed
        if self.scenario_parameters["number"] == 0:
            self.__generate_static_balloon_flights()
            self._balloon_status = np.ones(
                (self.balloon_parameters["num"], 1), dtype=int
            )
        else:
            self.__generate_balloon_flights()
            self._balloon_status = np.zeros(
                (self.balloon_parameters["num"], 1), dtype=int
            )

        self._balloon_states = self._balloon_flights[:, :, 0]
        self._rocket_sensors = np.full(12, np.nan)
        # Unusable steps per field, for rate limiting. Per field, because one
        # counter let a field that broke second go unnamed for the whole run.
        self._unusable_action_counts = Counter()
        self._rocket_states = np.full(13, np.nan)
        self._sweep_origin = None
        self.trajectories = None

        self.rocket_launched = False
        self.current_step = 0
        self.num_timesteps = self._balloon_flights.shape[2]
        self._popped_count = 0

        observation = self._get_obs()
        info = self._get_info()

        self.render_canvas = None
        self.render_balloons = None
        self.render_rocket = None
        self._render_frame()

        return observation, info

    def step(self, action):
        previous_balloon_positions = self._balloon_states[:, :3].copy()
        self.current_step += 1

        #  Update the balloon states
        self._balloon_states = self._balloon_flights[:, :, self.current_step]

        # Update balloon status: balloons that have reached the release step and are still on the ground become released
        released_mask = self.current_step >= self._balloon_release_at_step
        ground_mask = self._balloon_status[:, 0] == 0
        self._balloon_status[released_mask & ground_mask, 0] = 1

        # A diverging policy emits NaN, and it is the most ordinary failure there
        # is while training one. A step it cannot command is not a run it should
        # lose, so the command is dropped and the episode carries on.
        commands, unusable = _usable_action_fields(action)
        for field in sorted(unusable):
            # Rate limited per field, since a policy that breaks stays broken and
            # one line per step for the rest of the horizon is its own denial of
            # service. Per field, so the second one to break is still named.
            self._unusable_action_counts[field] += 1
            if self._unusable_action_counts[field] in _UNUSABLE_ACTION_LOG_STEPS:
                logger.warning(
                    "Step %d: ignoring %s, which the environment cannot use "
                    "(%d such steps for that field)",
                    self.current_step,
                    field,
                    self._unusable_action_counts[field],
                )

        if not self.rocket_launched:
            _rocket_finished = False
            # Refused rather than dropped, because there is no previous attitude
            # to fall back on. The agent can launch on any later step.
            if _wants_launch(commands) and "launch_inclination_heading" in commands:
                attitude = commands["launch_inclination_heading"]
                self.__get_init_rocket_states(attitude[0], attitude[1])
                self.initial_solution[0] = (
                    self.current_step * self.simulation_parameters["time_step"]
                )
                self.__init_rocket_simulation()
                self._sweep_origin = np.asarray(self.initial_solution[1:4], dtype=float)
                # Last, so a failure above leaves the environment unlaunched
                # rather than launched with no flight for the next step to use.
                self.rocket_launched = True
        else:  # Apply action to step the rocket simulation and get sensor measurements
            # Each actuator keeps the value it already holds when its field is
            # dropped, which is what a real one does between commands.
            if "roll" in commands:
                self._rocket_flight.rocket.roll_control.roll_torque = float(
                    commands["roll"][0]
                )
            if "tvc" in commands:
                thrust_vector = self._rocket_flight.rocket.thrust_vector_control
                thrust_vector.gimbal_angle_x = float(commands["tvc"][0])
                thrust_vector.gimbal_angle_y = float(commands["tvc"][1])
            if "throttle" in commands:
                self._rocket_flight.rocket.throttle_control.throttle = float(
                    commands["throttle"][0]
                )
            self._rocket_flight.step_simulation()
            _sensor = self._rocket_flight.sensors
            self._rocket_sensors[:3] = _sensor[0].measurement  # gyro
            self._rocket_sensors[3:6] = _sensor[1].measurement  # accel
            self._rocket_sensors[6:12] = _sensor[2].measurement  # gnss
            self._rocket_states = self._rocket_flight.y_sol[:]
            _rocket_finished = self._rocket_flight._step_state["finished"]

            # detect pops over the interval the rocket just flew
            self._detect_pops(previous_balloon_positions, self._sweep_origin)
            self._sweep_origin = self._rocket_states[:3].copy()

        # Append rocket and balloon states to trajectories for logging
        step_record = {
            "time": self.current_step * self.simulation_parameters["time_step"],
            "rocket_states": self._rocket_states.copy().tolist(),
            "balloon_states": self._balloon_states.copy().tolist(),
            "balloon_status": self._balloon_status[:, 0].tolist(),
        }
        if self.trajectories is None:
            self.trajectories = [step_record]
        else:
            self.trajectories.append(step_record)

        # An episode is done iff reaches max time or end of trajectory
        _timeout = self.current_step >= self.num_timesteps - 1
        if _timeout:
            logger.info("Truncated: Reached max time")
            # An agent is free never to launch, in which case there is no flight
            # to post-process and these calls would raise instead of ending the
            # episode.
            if self._rocket_flight is not None:
                self._rocket_flight.post_process_simulation()
                self._rocket_flight.initialize_prints_plots()
        elif _rocket_finished:
            logger.info("Terminated: Rocket flight finished")
        # Gymnasium keeps these apart on purpose. terminated means the MDP
        # reached a terminal state, which here is the flight ending. truncated
        # means something outside it stopped the episode, which is what running
        # out of precomputed horizon is. An algorithm bootstraps the value of
        # the final state when it was truncated and does not when it terminated,
        # so reporting the clock as termination teaches an agent that running
        # out of time is absorbing.
        terminated = _rocket_finished
        truncated = _timeout

        # Calculate reward based on newly popped balloons at this step
        new_count = np.sum(self._balloon_status[:, 0] == 2)
        reward = new_count - self._popped_count
        self._popped_count = new_count

        # Get observation and info for the current step
        observation = self._get_obs()
        info = self._get_info()

        # Render every 0.1 sec or on termination to balance visualization and performance
        _remainder = np.remainder(
            self.current_step, 0.1 / self.simulation_parameters["time_step"]
        )  # print every 0.1 sec
        # Both endings, not only termination. This used to read ``terminated``
        # while that flag covered the clock as well, so an episode ending on the
        # horizon always drew its last frame. Splitting the two causes quietly
        # took that away from every truncated episode whose final step does not
        # land on the 0.1 s cadence, which for scenario 1 is the usual ending.
        if _remainder == 0 or terminated or truncated:
            self._render_frame()

        return observation, reward, terminated, truncated, info

    @staticmethod
    def _segment_distance_squared_batch(
        segment_start_a,
        segment_end_a,
        segment_start_b,
        segment_end_b,
    ):
        """Return squared minimum distance between one segment and N segments."""
        segment_start_a = np.asarray(segment_start_a, dtype=float)
        segment_end_a = np.asarray(segment_end_a, dtype=float)
        segment_start_b = np.asarray(segment_start_b, dtype=float)
        segment_end_b = np.asarray(segment_end_b, dtype=float)

        # Two tolerances, because the quantities they guard are not the same
        # kind of thing. ``a_coeff`` and ``e_coeff`` are squared lengths in m**2,
        # so a length below a micron counts as no movement at all. The
        # denominator is a_coeff * e_coeff - b_coeff**2, in m**4, and how big it
        # is says nothing on its own: it depends on the segment lengths as much
        # as on the angle between them.
        #
        # One absolute value for both made the parallel decision depend on
        # scale. A 1 mm rocket segment and a 1 mm balloon segment at exactly
        # ninety degrees give a denominator of exactly 1e-12, so
        # ``> 1e-12`` called perpendicular segments parallel, took the branch
        # that pins s to zero, and reported 1.5005 m where the real distance is
        # 1.4995 m. With a 1.5 m radius that is a pop reported as a miss.
        #
        # Measured over a scenario-1 run: 56 of 9155 released-balloon
        # evaluations landed in that branch, all of them genuinely degenerate,
        # so no score moved. It is the shape of the test that is wrong rather
        # than any current result.
        degenerate_length_squared = 1e-12
        # Dimensionless: sin(angle)**2 has to clear this before the two
        # directions count as distinct, whatever the segments are scaled to.
        #
        # Derived from double precision rather than chosen. The denominator is
        # two nearly equal products subtracted, so its own rounding error is of
        # order eps * a_coeff * e_coeff, and a few multiples of that is the
        # point below which its sign and magnitude mean nothing. Eight is the
        # margin.
        #
        # Not a value to widen. The branch it selects pins s to zero, which is
        # the right answer only when the directions really are parallel, since
        # then every s gives the same distance. For merely close to parallel it
        # is wrong, and the wider the tolerance the more pairs land there. At
        # 1e-12 this pair returned 1.5000004 m where the true closest approach
        # is 1.4999995 m, which against a 1.5 m radius is a pop reported as a
        # miss:
        #
        #     rocket  (0, 0, 0)          -> (1, 0, 0)
        #     balloon (-1, 1.5000013, 0) -> (1, 1.4999995, 0)
        #
        # At 8 * eps that pair goes through the regular solution and is right.
        parallel_relative_epsilon = 8 * np.finfo(float).eps
        epsilon = degenerate_length_squared
        direction_a = segment_end_a - segment_start_a
        direction_b = segment_end_b - segment_start_b
        offset = segment_start_a - segment_start_b

        n_segments = segment_start_b.shape[0]
        s_param = np.zeros(n_segments)
        t_param = np.zeros(n_segments)

        a_coeff = float(np.dot(direction_a, direction_a))
        e_coeff = np.einsum("ij,ij->i", direction_b, direction_b)
        f_coeff = np.einsum("ij,ij->i", direction_b, offset)

        if a_coeff <= epsilon:
            valid_e = e_coeff > epsilon
            t_param[valid_e] = np.clip(
                f_coeff[valid_e] / e_coeff[valid_e],
                0.0,
                1.0,
            )
        else:
            c_coeff = np.einsum("j,ij->i", direction_a, offset)

            degenerate_b = e_coeff <= epsilon
            s_param[degenerate_b] = np.clip(
                -c_coeff[degenerate_b] / a_coeff,
                0.0,
                1.0,
            )

            regular = ~degenerate_b
            if np.any(regular):
                b_coeff = np.einsum("j,ij->i", direction_a, direction_b)
                denominator = a_coeff * e_coeff - b_coeff * b_coeff

                # Relative to a_coeff * e_coeff, which is what the denominator
                # would be at ninety degrees, so this compares sin(angle)**2
                # against a dimensionless tolerance and scaling both segments
                # cannot change the answer.
                non_parallel = regular & (
                    np.abs(denominator) > parallel_relative_epsilon * a_coeff * e_coeff
                )
                s_param[non_parallel] = np.clip(
                    (
                        b_coeff[non_parallel] * f_coeff[non_parallel]
                        - c_coeff[non_parallel] * e_coeff[non_parallel]
                    )
                    / denominator[non_parallel],
                    0.0,
                    1.0,
                )

                t_param[regular] = (
                    b_coeff[regular] * s_param[regular] + f_coeff[regular]
                ) / e_coeff[regular]

                t_too_low = regular & (t_param < 0.0)
                t_param[t_too_low] = 0.0
                s_param[t_too_low] = np.clip(
                    -c_coeff[t_too_low] / a_coeff,
                    0.0,
                    1.0,
                )

                t_too_high = regular & (t_param > 1.0)
                t_param[t_too_high] = 1.0
                s_param[t_too_high] = np.clip(
                    (b_coeff[t_too_high] - c_coeff[t_too_high]) / a_coeff,
                    0.0,
                    1.0,
                )

        closest_point_a = segment_start_a + s_param[:, None] * direction_a
        closest_point_b = segment_start_b + t_param[:, None] * direction_b
        separation = closest_point_a - closest_point_b
        return np.einsum("ij,ij->i", separation, separation)

    def _detect_pops(self, previous_balloon_positions, previous_rocket_position):
        """Detect pops using swept paths over the current timestep."""
        previous_balloon_positions = np.asarray(previous_balloon_positions, dtype=float)
        previous_rocket_position = np.asarray(previous_rocket_position, dtype=float)
        current_balloon_positions = np.asarray(self._balloon_states[:, :3], dtype=float)
        current_rocket_position = np.asarray(self._rocket_states[:3], dtype=float)
        balloon_radius_squared = self.balloon_parameters["radius"] ** 2
        released_mask = self._balloon_status[:, 0] == 1
        if not np.any(released_mask):
            return

        distance_squared = self._segment_distance_squared_batch(
            previous_rocket_position,
            current_rocket_position,
            previous_balloon_positions[released_mask],
            current_balloon_positions[released_mask],
        )
        popped_released = distance_squared <= balloon_radius_squared
        released_indices = np.flatnonzero(released_mask)
        self._balloon_status[released_indices[popped_released], 0] = 2

    def _render_frame(self):
        if self.render_mode == "vpython":
            from vpython import arrow, canvas, color, rate, sphere, vector

            if self.render_canvas is None:
                self.render_canvas = canvas(
                    title="Balloon Popping Environment",
                    width=800,
                    height=600,
                    center=vector(0, 0, 0),
                    background=color.white,
                )
                self.render_balloons = [
                    sphere(radius=1.5, color=color.magenta)
                    for _ in range(self.balloon_parameters["num"])
                ]
                # Create rocket arrow visualization
                self.render_rocket = arrow(
                    pos=vector(0, 0, 0),
                    axis=vector(0, 0, 5),
                    shaftwidth=0.5,
                    color=color.blue,
                )

            # Status colors: 0=grey (ground), 1=magenta (released), 2=red (popped)
            status_colors = {0: color.gray(0.5), 1: color.magenta, 2: color.red}
            for balloon, state, status in zip(
                self.render_balloons, self._balloon_states, self._balloon_status[:, 0]
            ):
                balloon.pos = vector(state[0], state[1], state[2])
                balloon.color = status_colors[int(status)]

            # Update rocket visualization with attitude
            if not np.isnan(self._rocket_states[0]):
                # Convert quaternion to rocket nose direction vector
                nose_direction = Matrix.transformation(
                    self._rocket_states[6:10]
                ) @ Vector([0, 0, 1])

                self.render_rocket.pos = vector(
                    self._rocket_states[0],
                    self._rocket_states[1],
                    self._rocket_states[2],
                )
                self.render_rocket.axis = vector(
                    nose_direction[0] * 10,
                    nose_direction[1] * 10,
                    nose_direction[2] * 10,
                )

            rate(30)
        elif self.render_mode == "matplotlib":
            if self.render_canvas is None:
                self.render_canvas = plt.figure().add_subplot(projection="3d")
                self.render_balloons = self.render_canvas.scatter(
                    self._balloon_states[:, 0],
                    self._balloon_states[:, 1],
                    self._balloon_states[:, 2],
                    c="magenta",
                )
                # Arrays, not scalars: plot documents xs/ys as array-like, and
                # the update below has to keep the same types (see #89).
                self.render_rocket = self.render_canvas.plot(
                    np.array([self._rocket_states[0]]),
                    np.array([self._rocket_states[1]]),
                    np.array([self._rocket_states[2]]),
                    "s",
                    color="blue",
                )
                self.render_canvas.set_xlabel("X position (m)")
                self.render_canvas.set_ylabel("Y position (m)")
                self.render_canvas.set_zlabel("Z position (m)")
                self.render_canvas.set_xlim(
                    self._balloon_flights[:, 0, :].min() - 10,
                    self._balloon_flights[:, 0, :].max() + 10,
                )
                self.render_canvas.set_ylim(
                    self._balloon_flights[:, 1, :].min() - 10,
                    self._balloon_flights[:, 1, :].max() + 10,
                )
                self.render_canvas.set_zlim(
                    0, self._balloon_flights[:, 2, :].max() + 10
                )

            # Update balloon positions and colors based on status (grey if ground, magenta if released, red if popped)
            status_colors = {0: "grey", 1: "magenta", 2: "red"}
            colors = [
                status_colors[int(status)] for status in self._balloon_status[:, 0]
            ]
            self.render_balloons._offsets3d = (
                self._balloon_states[:, 0],
                self._balloon_states[:, 1],
                self._balloon_states[:, 2],
            )
            self.render_balloons.set_facecolors(colors)
            # Arrays rather than lists. Line3D.draw reaches for `_verts3d[0].shape`
            # when a coordinate is invalid for the axis scale, and the rocket
            # state is all-NaN until the launch action builds the flight, so
            # every frame before launch takes that branch. With lists it raises
            # `'list' object has no attribute 'shape'` on Matplotlib 3.11 and
            # loses the run (#89). Finite coordinates never reach the branch,
            # which is why lists went unnoticed.
            self.render_rocket[0].set_data_3d(
                np.array([self._rocket_states[0]]),
                np.array([self._rocket_states[1]]),
                np.array([self._rocket_states[2]]),
            )
            self.render_canvas.set_title(
                f"Time: {self.current_step * self.simulation_parameters['time_step']:.2f} sec\nTotal Reward: {self._popped_count}"
            )
            plt.draw()
            plt.pause(0.001)
        else:
            pass

    def close(self):
        logger.debug("closing environment")

    def __create_environment(self):
        self._rocketpy_env = Environment(
            date=self.environment_parameters["date"],
            latitude=self.environment_parameters["latitude"],
            longitude=self.environment_parameters["longitude"],
            elevation=self.environment_parameters["elevation"],
            datum="WGS84",
            timezone="UTC",
        )
        if self.environment_parameters["atmosphere_data_filename"] is None:
            self._rocketpy_env.set_atmospheric_model(type="standard_atmosphere")
        else:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data",
                self.environment_parameters["atmosphere_data_filename"],
            )
            self._rocketpy_env.set_atmospheric_model(
                type="Ensemble",
                file=path,
                dictionary="ECMWF",
            )

        # Add gust after setting atmospheric model.
        if self.environment_parameters["gust"]["enable"]:
            gust_param = self.environment_parameters["gust"]

            altitude_nodes = np.arange(
                0.0,
                self._rocketpy_env.max_expected_height + gust_param["altitude_spacing"],
                gust_param["altitude_spacing"],
            )
            x_gust_nodes = self.np_random.uniform(
                -gust_param["max_gust_speed"],
                gust_param["max_gust_speed"],
                size=len(altitude_nodes),
            )
            y_gust_nodes = self.np_random.uniform(
                -gust_param["max_gust_speed"],
                gust_param["max_gust_speed"],
                size=len(altitude_nodes),
            )
            gust_decay = np.exp(
                -altitude_nodes / gust_param["gust_decay_height"]
            )  # Exponential decay of gust speed with altitude

            def gust_x(height_asl):
                # X direction = East
                return np.interp(
                    height_asl,
                    altitude_nodes,
                    x_gust_nodes * gust_decay,
                )

            def gust_y(height_asl):
                # Y direction = North
                return np.interp(
                    height_asl,
                    altitude_nodes,
                    y_gust_nodes * gust_decay,
                )

            self._rocketpy_env.add_wind_gust(gust_x, gust_y)

    def __reset_balloon_release_sequence(self):
        n = self.balloon_parameters["num"]
        i = self.balloon_parameters["release_interval"]
        t = self.simulation_parameters["time_step"]
        self._balloon_release_at_step = np.arange(n) * int(i / t)
        self._np_random.shuffle(self._balloon_release_at_step)

    def __generate_balloon_flights(self):
        monte_carlo_environment = copy.deepcopy(self._rocketpy_env)

        lat = self.environment_parameters["latitude"]
        lon = self.environment_parameters["longitude"]
        lat_std = self.balloon_parameters["stochastic"]["latitude_std"]
        lon_std = self.balloon_parameters["stochastic"]["longitude_std"]
        stochastic_env = StochasticEnvironment(
            environment=monte_carlo_environment,
            latitude=(
                lat - lat_std,
                lat + lat_std,
                "uniform",
            ),
            longitude=(
                lon - lon_std,
                lon + lon_std,
                "uniform",
            ),
        )

        SM = SolidMotor(
            thrust_source=50,
            burn_time=0.2,
            grain_number=1,
            grain_density=100,
            grain_initial_inner_radius=0.01,
            grain_outer_radius=0.035,
            grain_initial_height=0.1,
            nozzle_radius=0.0335,
            nozzle_position=0,
            throat_radius=0.0114,
            grain_separation=0.00,
            grains_center_of_mass_position=0.2,
            dry_inertia=(0, 0, 0),
            center_of_dry_mass_position=0,
            dry_mass=0,
        )

        cL0 = self.balloon_parameters["aero_coefficients"]["cL"]
        cQ0 = self.balloon_parameters["aero_coefficients"]["cQ"]
        cD0 = self.balloon_parameters["aero_coefficients"]["cD"]
        cDamping = self.balloon_parameters["aero_coefficients"]["moment_damping"]
        balloon_aero_model = LinearGenericSurface(
            reference_area=np.pi * self.balloon_parameters["radius"] ** 2,
            reference_length=1,
            coefficient_constants=[
                cL0, 0, 0, 0, 0, 0,    # cL_0, cL_alpha, cL_beta, cL_p, cL_q, cL_r
                cQ0, 0, 0, 0, 0, 0,    # cQ_0, cQ_alpha, cQ_beta, cQ_p, cQ_q, cQ_r
                cD0, 0, 0, 0, 0, 0,    # cD_0, cD_alpha, cD_beta, cD_p, cD_q, cD_r
                0, 0, 0, cDamping, cDamping, cDamping,   # cm_0, cm_alpha, cm_beta, cm_p, cm_q, cm_r
                0, 0, 0, cDamping, cDamping, cDamping,   # cn_0, cn_alpha, cn_beta, cn_p, cn_q, cn_r
                0, 0, 0, cDamping, cDamping, cDamping,   # cl_0, cl_alpha, cl_beta, cl_p, cl_q, cl_r
            ],
            center_of_pressure=(0, 0, 0),
            name="Balloon Aero Model",
        )  # fmt: skip

        Balloon = Rocket(
            volume=4 / 3 * np.pi * self.balloon_parameters["radius"] ** 3,
            radius=0.05,  # was designed for rocket, not balloon, hard code a value
            mass=self.balloon_parameters["mass"],
            inertia=self.balloon_parameters["inertia"],
            center_of_mass_without_motor=0.2,
            power_off_drag=0,
            power_on_drag=0,
            coordinate_system_orientation="tail_to_nose",
        )

        Balloon.add_motor(SM, position=0)
        Balloon.add_surfaces(balloon_aero_model, positions=(0, 0, 0.2))

        stochastic_balloon = StochasticRocket(
            rocket=Balloon,
            mass=self.balloon_parameters["stochastic"]["mass_std"],
            volume=self.balloon_parameters["stochastic"]["volume_std"],
            inertia_11=self.balloon_parameters["stochastic"]["inertia_std"],
            inertia_22=self.balloon_parameters["stochastic"]["inertia_std"],
            inertia_33=self.balloon_parameters["stochastic"]["inertia_std"],
            center_of_mass_without_motor=0,
        )
        stochastic_balloon.add_motor(SM, position=0)
        stochastic_balloon.add_linear_generic_surface(balloon_aero_model)

        flight = Flight(
            rocket=Balloon,
            environment=monte_carlo_environment,
            inclination=90,
            heading=180,
            rail_length=0.1,
            max_time=self.simulation_parameters["max_time"],
            verbose=True,
            run_simulation=False,
            ode_solver="RK45",
        )
        stochastic_flight = StochasticFlight(
            flight=flight,
            inclination=5,
            heading=90,
        )

        time_array = np.arange(
            0,
            self.simulation_parameters["max_time"],
            self.simulation_parameters["time_step"],
        )
        # MonteCarlo writes .inputs.txt, .outputs.txt and .errors.txt beside its
        # filename, and for scenario 1 the outputs file is 169 MB. A successful
        # reset reads none of them back: the arrays below come from the returned
        # dict. Written straight into the shared temp directory they were never
        # removed, so a competitor iterating on an agent left one set behind per
        # run and filled /tmp within tens of runs, after which runs fail in ways
        # that do not point back here. Measured: 22 files, 3.7 GB, and one
        # corrupted run.
        #
        # Removed only once the whole conversion below has succeeded, which is
        # why this is not a TemporaryDirectory. RocketPy keeps the failing
        # stochastic inputs on purpose: a serial simulation error appends them to
        # .errors.txt and re-raises, and Ctrl-C prints "Files saved." A context
        # manager deletes the directory while the exception unwinds, so the
        # caller is told the files were saved and finds nothing there. The
        # conversion is inside the same block because it has its own ways to
        # fail, and an incomplete result is exactly when the inputs that produced
        # it are worth having.
        #
        # mkdtemp creates the directory exclusively, so this also subsumes the
        # per-process filename that was there to keep two runs apart.
        monte_carlo_dir = tempfile.mkdtemp(prefix="balloon_sim_")
        try:
            monte_carlo_sim = MonteCarlo(
                filename=os.path.join(monte_carlo_dir, "balloon_sim"),
                environment=stochastic_env,
                rocket=stochastic_balloon,
                flight=stochastic_flight,
                export_list=["t_final"],
                data_collector={
                    "x": lambda flight: flight.x(time_array),
                    "y": lambda flight: flight.y(time_array),
                    "z": lambda flight: flight.z(time_array),
                    "vx": lambda flight: flight.vx(time_array),
                    "vy": lambda flight: flight.vy(time_array),
                    "vz": lambda flight: flight.vz(time_array),
                    "lat0": lambda flight: flight.latitude(0),
                    "lon0": lambda flight: flight.longitude(0),
                },
            )

            # One read, used for both the request and the check on what came
            # back, so the two cannot drift apart later.
            requested_balloons = self.balloon_parameters["num"]
            monte_carlo_results_ = monte_carlo_sim.simulate(
                number_of_simulations=requested_balloons,
                append=False,
                include_function_data=False,
                random_seed=self.np_random_seed,
                parallel=False,
            )

            _check_monte_carlo_returned_every_balloon(
                monte_carlo_results_, requested_balloons
            )

            # Convert Monte Carlo dict to [balloon][state][timestep].
            east0, north0, up0 = pm.geodetic2enu(
                monte_carlo_results_["lat0"],
                monte_carlo_results_["lon0"],
                self._rocketpy_env.elevation,
                self._rocketpy_env.latitude,
                self._rocketpy_env.longitude,
                self._rocketpy_env.elevation,
            )
            # Broadcast initial ENU offsets to all timesteps for each simulation
            east0, north0, up0 = (
                np.array(east0)[:, None],
                np.array(north0)[:, None],
                np.array(up0)[:, None],
            )

            self._balloon_flights = np.stack(
                [
                    np.array(monte_carlo_results_["x"]) + east0,
                    np.array(monte_carlo_results_["y"]) + north0,
                    np.array(monte_carlo_results_["z"]) + up0,
                    np.array(monte_carlo_results_["vx"]),
                    np.array(monte_carlo_results_["vy"]),
                    np.array(monte_carlo_results_["vz"]),
                ],
                axis=1,
            )
        except BaseException:
            # Including KeyboardInterrupt, which is the case RocketPy prints
            # "Files saved." for. Keeping them costs one directory; removing
            # them costs whoever has to work out why a run failed.
            #
            # Only when there is something to keep. A failure before any file
            # was written, which is every failure inside the MonteCarlo
            # constructor, would otherwise leave an empty directory behind on
            # every attempt: the leak this whole change is about, in a smaller
            # size.
            if _monte_carlo_workspace_has_files(monte_carlo_dir):
                logger.warning(
                    "Balloon Monte Carlo failed; its inputs and error log are at %s",
                    monte_carlo_dir,
                )
            else:
                _remove_monte_carlo_workspace(monte_carlo_dir)
            raise
        _remove_monte_carlo_workspace(monte_carlo_dir)

        # Vectorized shift trajectories by release step
        num_balloons, state_dims, num_timesteps = self._balloon_flights.shape
        release_steps = np.asarray(self._balloon_release_at_step, dtype=int)
        release_steps = np.clip(release_steps, 0, num_timesteps)

        # Create source time indices for each balloon
        time_idx = np.arange(num_timesteps)
        source_idx = np.clip(
            time_idx[np.newaxis, :] - release_steps[:, np.newaxis],
            0,
            num_timesteps - 1,
        )

        # Gather shifted trajectories using proper advanced indexing
        # Create indices that broadcast to (num_balloons, state_dims, num_timesteps)
        balloon_idx = np.arange(num_balloons)[:, None, None]  # (num_balloons, 1, 1)
        state_idx = np.arange(state_dims)[None, :, None]  # (1, state_dims, 1)
        shifted = self._balloon_flights[balloon_idx, state_idx, source_idx[:, None, :]]

        # Fill pre-release timesteps with initial state
        pre_release_mask = (
            time_idx < release_steps[:, np.newaxis]
        )  # (num_balloons, num_timesteps)
        initial_states = self._balloon_flights[
            :, :, 0:1
        ]  # (num_balloons, state_dims, 1)

        # Apply mask: for pre-release times, use initial state; otherwise use shifted
        pre_release_mask_expanded = pre_release_mask[
            :, np.newaxis, :
        ]  # (num_balloons, 1, num_timesteps)
        shifted = np.where(pre_release_mask_expanded, initial_states, shifted)

        self._balloon_flights = shifted

    def __generate_static_balloon_flights(self):
        """Scenario 0: balloons sit at fixed heights, so no Monte Carlo is run.

        Builds the [balloon, state, timestep] array directly: x, y and all
        velocities stay 0; z is a per-balloon constant spaced 40 m apart.
        """
        num_balloons = self.balloon_parameters["num"]
        num_timesteps = len(
            np.arange(
                0,
                self.simulation_parameters["max_time"],
                self.simulation_parameters["time_step"],
            )
        )
        self._balloon_flights = np.zeros((num_balloons, 6, num_timesteps))
        z_values = 10 + self._rocketpy_env.elevation + np.arange(num_balloons) * 40
        self._balloon_flights[:, 2, :] = z_values[:, None]

    def __get_init_rocket_states(self, inclination, heading):
        # Initialize time and state variables
        t_initial = 0
        x_init, y_init, z_init = 0, 0, self.environment_parameters["elevation"]
        vx_init, vy_init, vz_init = 0, 0, 0
        w1_init, w2_init, w3_init = 0, 0, 0

        # Initialize attitude
        e0_init, e1_init, e2_init, e3_init = get_initial_attitude(inclination, heading)

        # Store initial conditions
        self.initial_solution = [
            t_initial,
            x_init,
            y_init,
            z_init,
            vx_init,
            vy_init,
            vz_init,
            e0_init,
            e1_init,
            e2_init,
            e3_init,
            w1_init,
            w2_init,
            w3_init,
        ]

    def __init_rocket_simulation(self):
        # Rocket flight simulation initialization

        # Create tank fluids from parameters
        tank_cfg = self.rocket_parameters["tank"]
        oxidizer_gas = Fluid(name=tank_cfg["gas"], density=tank_cfg["gas_density"])
        oxidizer_liq = Fluid(
            name=tank_cfg["liquid"], density=tank_cfg["liquid_density"]
        )

        # Create tank from parameters
        tank_shape = CylindricalTank(
            radius=tank_cfg["radius"], height=tank_cfg["height"]
        )
        oxidizer_tank = MassFlowRateBasedTank(
            name="oxidizer_tank",
            geometry=tank_shape,
            flux_time=(
                self.initial_solution[0],
                self.initial_solution[0] + tank_cfg["flux_time"],
            ),
            initial_liquid_mass=tank_cfg["initial_liquid_mass"],
            initial_gas_mass=tank_cfg["initial_gas_mass"],
            liquid_mass_flow_rate_in=0,
            liquid_mass_flow_rate_out=tank_cfg["liquid_mass_flow_rate_out"],
            gas_mass_flow_rate_in=0,
            gas_mass_flow_rate_out=0,
            liquid=oxidizer_liq,
            gas=oxidizer_gas,
        )

        # Create motor from parameters
        motor_cfg = self.rocket_parameters["motor"]
        hybrid_motor = HybridMotor(
            thrust_source=motor_cfg["thrust_source"],
            dry_mass=motor_cfg["dry_mass"],
            dry_inertia=tuple(motor_cfg["dry_inertia"]),
            center_of_dry_mass_position=motor_cfg["center_of_dry_mass_position"],
            burn_time=(
                self.initial_solution[0],
                motor_cfg["burn_time"] + self.initial_solution[0],
            ),
            reshape_thrust_curve=False,
            grain_number=motor_cfg["grain_number"],
            grain_separation=motor_cfg["grain_separation"],
            grain_outer_radius=motor_cfg["grain_outer_radius"],
            grain_initial_inner_radius=motor_cfg["grain_initial_inner_radius"],
            grain_initial_height=motor_cfg["grain_initial_height"],
            grain_density=motor_cfg["grain_density"],
            nozzle_radius=motor_cfg["nozzle_radius"],
            throat_radius=motor_cfg["throat_radius"],
            interpolation_method="linear",
            nozzle_position=motor_cfg["nozzle_position"],
            grains_center_of_mass_position=motor_cfg["grains_center_of_mass_position"],
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )

        # Add tank to motor
        hybrid_motor.add_tank(tank=oxidizer_tank, position=tank_cfg["tank_position"])

        # Create rocket body from parameters
        rocket_cfg = self.rocket_parameters["rocket_body"]
        rocket = Rocket(
            radius=rocket_cfg["radius"],
            mass=rocket_cfg["mass"],
            inertia=tuple(rocket_cfg["inertia"]),
            center_of_mass_without_motor=rocket_cfg["center_of_mass_without_motor"],
            power_off_drag=rocket_cfg["power_off_drag"],
            power_on_drag=rocket_cfg["power_on_drag"],
            coordinate_system_orientation="tail_to_nose",
            volume=rocket_cfg["volume"],
        )

        # Add motor from parameters
        rocket.add_motor(hybrid_motor, position=motor_cfg["motor_position"])

        # Add nose from parameters
        nose_cfg = self.rocket_parameters["nose"]
        rocket.add_nose(
            length=nose_cfg["length"],
            kind=nose_cfg["kind"],
            position=nose_cfg["position"],
        )

        # Add fins from parameters
        fins_cfg = self.rocket_parameters["fins"]
        if fins_cfg["useFins"]:
            rocket.add_trapezoidal_fins(
                n=fins_cfg["n"],
                span=fins_cfg["span"],
                root_chord=fins_cfg["root_chord"],
                tip_chord=fins_cfg["tip_chord"],
                position=fins_cfg["position"],
            )

        # Add sensors from parameters
        sensors_cfg = self.rocket_parameters["sensors"]
        # Seed each sensor's noise generator deterministically from the scenario
        # seed, so the sensor-noise realization is reproducible and independent
        # of the process-global RNG and of how the run is parallelized (the agent
        # never sees the seed). Distinct sub-streams decorrelate the three
        # sensors; the domain tag keeps them clear of the balloon Monte Carlo's
        # own SeedSequence(scenario_seed) tree even once that runs in parallel.
        sensor_seed_domain = 0x5E2502  # fixed "sensor" tag, distinct from the MC
        seed = self.np_random_seed
        if seed is None or seed < 0:
            # Gymnasium reports -1 when the seed is unknown, which happens when
            # np_random was assigned directly, and its property initializes
            # rather than ever handing back None. Both land here because
            # SeedSequence rejects any negative entropy, not only -1, so
            # narrowing this to == -1 would turn an unexpected value into a
            # crash. The entropy comes from the generator we do have instead of
            # a seed we do not; leaving the sensors unseeded here would make a
            # run unreproducible without saying so.
            entropy = [
                int(word)
                for word in self.np_random.integers(0, 2**32, size=4, dtype=np.uint32)
            ]
        else:
            entropy = [int(seed)]
        # Plain ints, not the SeedSequence children themselves: sensors keep
        # their seed and hand it back through to_dict(), which has to stay
        # JSON serializable for the submission packer.
        gyro_seed, accelerometer_seed, gnss_seed = (
            _seed_sequence_to_int(child)
            for child in np.random.SeedSequence([*entropy, sensor_seed_domain]).spawn(3)
        )
        gyro = Gyroscope(
            sampling_rate=sensors_cfg["sampling_rate"],
            noise_density=sensors_cfg["gyro_noise_density"],
            random_walk_density=sensors_cfg["gyro_random_walk_density"],
            constant_bias=sensors_cfg["gyro_constant_bias"],
            seed=gyro_seed,
        )
        accelerometer = Accelerometer(
            sampling_rate=sensors_cfg["sampling_rate"],
            noise_density=sensors_cfg["accelerometer_noise_density"],
            random_walk_density=sensors_cfg["accelerometer_random_walk_density"],
            constant_bias=sensors_cfg["accelerometer_constant_bias"],
            consider_gravity=True,
            seed=accelerometer_seed,
        )
        gnss = GnssReceiver(
            sampling_rate=sensors_cfg["sampling_rate"],
            position_accuracy=sensors_cfg["gnss_position_accuracy"],
            altitude_accuracy=sensors_cfg["gnss_altitude_accuracy"],
            velocity_accuracy=sensors_cfg["gnss_velocity_accuracy"],
            seed=gnss_seed,
        )
        rocket.add_sensor(gyro, position=sensors_cfg["gyro_position"])
        rocket.add_sensor(accelerometer, position=sensors_cfg["accelerometer_position"])
        rocket.add_sensor(gnss, position=sensors_cfg["gnss_position"])
        # rocket.draw()

        # Add control systems from parameters
        control_cfg = self.rocket_parameters["control"]

        def tvc_controller_function(
            time, sampling_rate, state, state_history, observed_variables, tvc, sensors
        ):
            # log tvc angles
            return (
                time,
                tvc.gimbal_angle_x,
                tvc.gimbal_angle_y,
            )

        rocket.add_thrust_vector_control(
            controller_function=tvc_controller_function,
            sampling_rate=1 / self.simulation_parameters["time_step"],
            max_gimbal_angle=control_cfg["max_gimbal_angle"],
            gimbal_rate_limit=control_cfg["gimbal_rate_limit"],
            gimbal_time_constant=control_cfg["gimbal_time_constant"],
        )

        def roll_controller_function(
            time,
            sampling_rate,
            state,
            state_history,
            observed_variables,
            roll_control,
            sensors,
        ):
            # log roll control torques
            return (
                time,
                roll_control.roll_torque,
            )

        rocket.add_roll_control(
            controller_function=roll_controller_function,
            sampling_rate=1 / self.simulation_parameters["time_step"],
            max_roll_torque=control_cfg["max_roll_torque"],
            torque_rate_limit=control_cfg["torque_rate_limit"],
            roll_torque_time_constant=control_cfg["roll_torque_time_constant"],
        )

        def throttle_controller_function(
            time,
            sampling_rate,
            state,
            state_history,
            observed_variables,
            throttle_control,
            sensors,
        ):
            # log throttle
            return (
                time,
                throttle_control.throttle,
            )

        rocket.add_throttle_control(
            controller_function=throttle_controller_function,
            sampling_rate=1 / self.simulation_parameters["time_step"],
            throttle_range=control_cfg["throttle_range"],
            throttle_rate_limit=control_cfg["throttle_rate_limit"],
            throttle_time_constant=control_cfg["throttle_time_constant"],
        )

        self._rocket_flight = Flight(
            rocket=rocket,
            environment=self._rocketpy_env,
            rail_length=0.01,  # No rail since we directly set initial conditions to simulate launch
            initial_solution=self.initial_solution,
            max_time=self.simulation_parameters["max_time"],
            time_overshoot=False,
            verbose=False,
            run_simulation=False,
            ode_solver="RK45",
        )


# Helper function to convert inclination and heading to initial attitude quaternions
def get_initial_attitude(inclination, heading):
    # Precession / Heading Angle
    psi_init = np.radians(-heading)
    # Nutation / Attitude Angle
    theta_init = np.radians(inclination - 90)
    # Spin / Bank Angle
    phi_init = 0

    # 3-1-3 Euler Angles to Euler Parameters
    e0_init, e1_init, e2_init, e3_init = euler313_to_quaternions(
        phi_init, theta_init, psi_init
    )
    return e0_init, e1_init, e2_init, e3_init
