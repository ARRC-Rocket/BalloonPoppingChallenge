"""Integration test: sensor-noise determinism in the environment.

The competition's fairness model requires that, for a given announced scenario
seed, every competitor sees the *same* noisy sensor realization, reproducibly,
regardless of the process-global RNG state or how the run is parallelized (the
agent never sees the seed). Sensor noise is now drawn from a per-sensor
generator seeded from ``scenario.random_seed`` (decorrelated sub-streams), not
the process-global ``numpy.random``.

Scenarios #0/#1 ship with ideal (zero-noise) sensors, so this test switches on
non-zero noise to actually exercise the sensor RNG, then checks reproducibility.
Importing the env pulls in the simulation stack, so the test skips when that
stack is absent (mirroring the rest of the suite).
"""

import unittest
from importlib.util import find_spec

import numpy as np

# Only the simulation stack is optional. Guarding this package's own imports too
# would turn a renamed symbol or a broken module into a silent skip.
_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.agents.example_agents import AttitudeRateControlAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

# Launch almost immediately and run well past launch so the sensors (sampled at
# 100 Hz with time_step 0.01 s) actually produce noisy readings to compare.
AGENT_KWARGS = {"rate_targets": [0.0, 0.0, 0.0], "launch_time": 0.1}
N_STEPS = 70


def _noisy_scenario_0():
    """Scenario #0 with non-zero sensor noise so the sensor RNG is exercised."""
    scenario_params, given_params = load_scenario_parameters(0)
    sensors = scenario_params["rocket"]["sensors"]
    sensors["gyro_noise_density"] = 0.01
    sensors["accelerometer_noise_density"] = 0.05
    sensors["gnss_position_accuracy"] = 2.0
    sensors["gnss_altitude_accuracy"] = 2.0
    sensors["gnss_velocity_accuracy"] = 0.5
    return scenario_params, given_params


def _sensor_stream(scenario_params, given_params, seed, n_steps=N_STEPS):
    """Run a short episode and return the per-step rocket_sensors observations."""
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
    agent = AttitudeRateControlAgent(given_params, **AGENT_KWARGS)
    observation, _ = env.reset(seed=seed)
    readings = []
    terminated = False
    steps = 0
    while not terminated and steps < n_steps:
        observation, _, terminated, _, _ = env.step(agent.get_action(observation))
        readings.append(np.asarray(observation["rocket_sensors"], dtype=float))
        steps += 1
    return np.array(readings)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestSensorDeterminism(unittest.TestCase):
    """Seeded sensor noise must be reproducible and global-RNG-independent."""

    def test_same_seed_reproduces_sensor_noise(self):
        scenario_params, given_params = _noisy_scenario_0()
        first = _sensor_stream(scenario_params, given_params, seed=0)
        second = _sensor_stream(scenario_params, given_params, seed=0)
        self.assertTrue(
            np.array_equal(first, second, equal_nan=True),
            "same scenario seed must reproduce the sensor-noise stream exactly",
        )

    def test_different_seeds_change_sensor_noise(self):
        # Guards against a vacuous pass: with non-zero noise, two seeds must
        # produce different sensor readings, proving the RNG was exercised.
        scenario_params, given_params = _noisy_scenario_0()
        first = _sensor_stream(scenario_params, given_params, seed=0)
        second = _sensor_stream(scenario_params, given_params, seed=1)
        self.assertFalse(
            np.array_equal(first, second, equal_nan=True),
            "different seeds must yield different sensor noise",
        )

    def test_sensor_noise_independent_of_global_numpy_rng(self):
        # The fairness-critical guard: perturbing the process-global RNG between
        # runs must not change a seeded run's sensor noise.
        scenario_params, given_params = _noisy_scenario_0()
        np.random.seed(12345)
        first = _sensor_stream(scenario_params, given_params, seed=0)
        np.random.seed(67890)
        _ = np.random.random(1000)
        second = _sensor_stream(scenario_params, given_params, seed=0)
        self.assertTrue(
            np.array_equal(first, second, equal_nan=True),
            "seeded sensor noise must not depend on the process-global RNG",
        )


if __name__ == "__main__":
    unittest.main()
