from enum import Enum
from unittest import case
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rocketpy import (
    Environment,
    Flight,
    Function,
    LinearGenericSurface,
    MonteCarlo,
    Rocket,
    SolidMotor,
    StochasticEnvironment,
    StochasticFlight,
    StochasticRocket,
)
from vpython import canvas, color, vector, rate, sphere, arrow
import matplotlib.pyplot as plt

class BalloonPoppingEnv(gym.Env):
    metadata = {"render_modes": ["vpython", "matplotlib"]}
    def __init__(self, settings):

        self.environment_settings = settings["environment"]
        self.simulation_settings = settings["simulation"]
        self.balloon_settings = settings["balloon"]
        self.rocket_settings = settings["rocket"]

        self._balloon_paths = None
        self._balloon_states = np.array(np.zeros((self.balloon_settings["num"], 6)))    # [balloon_num, (x, y, z, vx, vy, vz)]
        self._rocket_states = np.array(np.zeros(12))                                    # (gyroX, gyroY, gyroZ, accX, accY, accZ, posX, posY, posZ, velX, velY, velZ)

        # Observations include balloon and rocket states
        self.observation_space = spaces.Dict(
            {
                "balloon": spaces.Box(low=-np.inf*np.ones((self.balloon_settings["num"], 6)), high=np.inf*np.ones((self.balloon_settings["num"], 6)), dtype=np.float64),
                "rocket": spaces.Box(low=-np.inf*np.ones(12), high=np.inf*np.ones(12), dtype=np.float64),
            }
        )

        # TVC, roll, and throttling actions
        self.action_space = spaces.Dict(
            {
                "TVC":  spaces.Box(low=-self.rocket_settings["tvc_gimbal_range"]*np.ones(2), high=self.rocket_settings["tvc_gimbal_range"]*np.ones(2), dtype=np.float64),
                "throttle": spaces.Box(low=self.rocket_settings["throttle_range"][0], high=self.rocket_settings["throttle_range"][1], shape=(1,), dtype=np.float64),
                "roll": spaces.Box(low=-self.rocket_settings["roll_range"]*np.ones(2), high=self.rocket_settings["roll_range"]*np.ones(2), dtype=np.float64),
            }
        )

        # Graphics-related attributes
        self.render_mode = settings["render"]["mode"]
        self.balloons = None
        self.canvas = None

    def _get_obs(self):
        return {"balloon": self._balloon_states, "rocket": self._rocket_states}

    def _get_info(self):
        return {"balloon_altitude": self._balloon_states[:, 2]}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_step = 0

        self._balloon_paths = generate_balloon_flights(self.environment_settings, self.simulation_settings, self.balloon_settings)

        self.num_timesteps = self._balloon_paths.shape[2]
        self._balloon_states = self._balloon_paths[:, :, self.current_step]
        self._rocket_states = np.array(np.zeros(12))

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1

        self._balloon_states = self._balloon_paths[:, :, self.current_step]
        self._rocket_states = np.array(np.zeros(12))

        # An episode is done iff reaches max time or end of trajectory
        terminated = (self.current_step >= self.num_timesteps - 1)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        if self.render_mode == "vpython":
            if self.canvas is None:
                self.canvas = canvas(title="Balloon Popping Environment", width=800, height=600, center=vector(0,0,0), background=color.white)
                self.balloons = sphere(radius = 1.5, color = color.magenta, make_trail=True)

            self.balloons.pos = vector(self._balloon_states[0, 0], self._balloon_states[0, 1], self._balloon_states[0, 2])
            rate(30)
        elif self.render_mode == "matplotlib":
            if self.canvas is None:
                self.canvas = plt.figure().add_subplot(projection='3d')
                self.balloons, = self.canvas.plot(self._balloon_states[:, 0], self._balloon_states[:, 1], self._balloon_states[:, 2], 'o', color='magenta')
                self.canvas.set_xlabel('X position (m)')
                self.canvas.set_ylabel('Y position (m)')
                self.canvas.set_zlabel('Z position (m)')
                self.canvas.set_xlim(-1000, 1000)
                self.canvas.set_ylim(-1000, 1000)
                self.canvas.set_zlim(0, 1000)

            if np.remainder(self.current_step,1/self.simulation_settings['time_step']) == 0:
                self.balloons.set_data(self._balloon_states[:, 0], self._balloon_states[:, 1])
                self.balloons.set_3d_properties(self._balloon_states[:, 2])
                self.canvas.set_title(f"Time: {self.current_step*self.simulation_settings['time_step']} sec")
                plt.draw()
                plt.pause(0.01)

    def close(self):
        print('closing environment')

def generate_balloon_flights(environment_settings, simulation_settings, balloon_settings):
    env = Environment(
        date=environment_settings["date"],
        latitude=environment_settings["latitude"],
        longitude=environment_settings["longitude"],
        elevation=environment_settings["elevation"],
        datum="WGS84",
        timezone="UTC",
    )
    env.set_atmospheric_model(
        type="Ensemble",
        file=environment_settings["atmosphere_data_path"],
        dictionary="ECMWF",
    )
    # env.max_expected_height = 4000

    stochastic_env = StochasticEnvironment(
        environment=env,
    )
    stochastic_env.visualize_attributes()

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

    cL0 = balloon_settings["aero_coefficients"]["cL"]
    cQ0 = balloon_settings["aero_coefficients"]["cQ"]
    cD0 = balloon_settings["aero_coefficients"]["cD"]
    cDamping = balloon_settings["aero_coefficients"]["moment_damping"]
    balloon_aero_model = LinearGenericSurface(
        reference_area=np.pi * balloon_settings["radius"] ** 2,
        reference_length=1,
        coefficient_constants=[
            cL0, 0, 0, 0, 0, 0,    # cL_0, cL_alpha, cL_beta, cL_p, cL_q, cL_r
            cQ0, 0, 0, 0, 0, 0,    # cQ_0, cQ_alpha, cQ_beta, cQ_p, cQ_q, cQ_r
            cD0, 0, 0, 0, 0, 0,    # cD_0, cD_alpha, cD_beta, cD_p, cD_q, cD_r
            0, 0, 0, cDamping, cDamping, cDamping,   # cm_0, cm_alpha, cm_beta, cm_p, cm_q, cm_r
            0, 0, 0, cDamping, cDamping, cDamping,   # cn_0, cn_alpha, cn_beta, cn_p, cn_q, cn_r
            0, 0, 0, cDamping, cDamping, cDamping,   # cl_0, cl_alpha, cl_beta, cl_p, cl_q, cl_r
            ],
        center_of_pressure=(0,0,0),
        name="Balloon Aero Model"
    )

    Balloon = Rocket(
        volume= 4 / 3 * np.pi * balloon_settings["radius"] ** 3,
        radius=0.05,
        mass=balloon_settings["mass"],
        inertia=balloon_settings["inertia"],
        center_of_mass_without_motor=0.2,
        power_off_drag=0,
        power_on_drag=0,
        coordinate_system_orientation="tail_to_nose",
    )

    Balloon.add_motor(SM, position=0)
    Balloon.add_surfaces(balloon_aero_model, positions=(0, 0, 0.2))
    Balloon.prints.rocket_aerodynamics_quantities()

    stochastic_balloon = StochasticRocket(
        rocket=Balloon,
        mass=balloon_settings["stochastic"]["mass"],
        volume=balloon_settings["stochastic"]["volume"],
        inertia_11=balloon_settings["stochastic"]["inertia"],
        inertia_22=balloon_settings["stochastic"]["inertia"],
        inertia_33=balloon_settings["stochastic"]["inertia"],
        center_of_mass_without_motor=0,
    )
    stochastic_balloon.add_motor(SM, position=0)
    stochastic_balloon.add_linear_generic_surface(balloon_aero_model)
    stochastic_balloon.visualize_attributes()

    flight = Flight(
        rocket=Balloon,
        environment=env,
        inclination=90,
        heading=180,
        rail_length=0.1,
        max_time=simulation_settings["max_time"],
        # max_time_step=0.01,
        # min_time_step=0.01,
        verbose=True,
    )
    stochastic_flight = StochasticFlight(
        flight=flight,
        inclination=5,
        heading=90,
    )
    stochastic_flight.visualize_attributes()

    monte_carlo_sim = MonteCarlo(
        filename="./BalloonPoppingGymEnv/envs/data/balloon_sim",
        environment=stochastic_env,
        rocket=stochastic_balloon,
        flight=stochastic_flight,
        export_list=["t_final"],
        data_collector={
            "x": lambda flight: flight.x(np.arange(0, simulation_settings["max_time"], simulation_settings["time_step"])),
            "y": lambda flight: flight.y(np.arange(0, simulation_settings["max_time"], simulation_settings["time_step"])),
            "z": lambda flight: flight.z(np.arange(0, simulation_settings["max_time"], simulation_settings["time_step"])),
            "vx": lambda flight: flight.vx(np.arange(0, simulation_settings["max_time"], simulation_settings["time_step"])),
            "vy": lambda flight: flight.vy(np.arange(0, simulation_settings["max_time"], simulation_settings["time_step"])),
            "vz": lambda flight: flight.vz(np.arange(0, simulation_settings["max_time"], simulation_settings["time_step"])),
        },
    )

    monte_carlo_results_ = monte_carlo_sim.simulate(
        number_of_simulations=balloon_settings["num"],
        append=False,
        include_function_data=False,
        parallel=False,
        n_workers=4,
    )

    """Convert Monte Carlo dict to [balloon][state][timestep]."""
    states = ["x", "y", "z", "vx", "vy", "vz"]
    monte_carlo_results = [np.array(monte_carlo_results_[state]) for state in states]
    return np.stack(monte_carlo_results, axis=1)