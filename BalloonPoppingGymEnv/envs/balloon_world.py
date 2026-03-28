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
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank

from vpython import canvas, color, vector, rate, sphere, arrow
import matplotlib.pyplot as plt
import pymap3d as pm

from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gyroscope import Gyroscope
from rocketpy.sensors.gnss_receiver import GnssReceiver

class BalloonPoppingEnv(gym.Env):
    metadata = {"render_modes": ["vpython", "matplotlib"]}
    def __init__(self, settings):

        self.environment_settings = settings["environment"]
        self.simulation_settings = settings["simulation"]
        self.balloon_settings = settings["balloon"]
        self.rocket_settings = settings["rocket"]

        self._balloon_status = np.array(np.zeros((self.balloon_settings["num"], 1)))    # [balloon_num, status] status: 0-ground; 1-released; 2-popped 
        self._balloon_states = np.array(np.zeros((self.balloon_settings["num"], 6)))    # [balloon_num, (x, y, z, vx, vy, vz)]
        self._rocket_sensors = np.full(12, np.nan)                                      # (gyroX, gyroY, gyroZ, accX, accY, accZ, posX, posY, posZ, velX, velY, velZ)
        self._rocket_states = np.full(13, np.nan)                                       # (posX, posY, posZ, velX, velY, velZ, e0, e1, e2, e3, w1, w2, w3)

        # Observations include balloon and rocket states
        self.observation_space = spaces.Dict(
            {
                "balloon_time": spaces.Box(low=0, high=self.simulation_settings["max_time"], shape=(1,), dtype=np.float64),
                "balloon_status": spaces.MultiDiscrete(3*np.ones((self.balloon_settings["num"], 1))),
                "balloon_states": spaces.Box(low=-np.inf*np.ones((self.balloon_settings["num"], 6)), high=np.inf*np.ones((self.balloon_settings["num"], 6)), dtype=np.float64),
                "rocket_time": spaces.Box(low=0, high=self.simulation_settings["max_time"], shape=(1,), dtype=np.float64),
                "rocket_sensors": spaces.Box(low=-np.inf*np.ones(12), high=np.inf*np.ones(12), dtype=np.float64),
            }
        )

        # tvc, roll, and throttling actions
        self.action_space = spaces.Dict(
            {
                "launch": spaces.MultiBinary(1),
                "tvc":  spaces.Box(low=-self.rocket_settings["tvc_gimbal_range"]*np.ones(2), high=self.rocket_settings["tvc_gimbal_range"]*np.ones(2), dtype=np.float64),
                "throttle": spaces.Box(low=self.rocket_settings["throttle_range"][0], high=self.rocket_settings["throttle_range"][1], shape=(1,), dtype=np.float64),
                "roll": spaces.Box(low=-self.rocket_settings["max_roll_torque"], high=self.rocket_settings["max_roll_torque"], dtype=np.float64),
            }
        )

        # Graphics-related attributes
        self.render_mode = settings["render"]["mode"]
        self.render_canvas = None
        self.render_balloons = None
        self.render_rocket = None

    def _get_obs(self):
        return {
            "balloon_time": self.current_step*self.simulation_settings['time_step'],
            "balloon_status": self._balloon_status,
            "balloon_states": self._balloon_states,
            "rocket_time": self._rocket_flight.t,
            "rocket_sensors": self._rocket_sensors
        }

    def _get_info(self):
        return {"rocket_states": self._rocket_states}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_step = 0

        self._balloon_flights = generate_balloon_flights(self.environment_settings, self.simulation_settings, self.balloon_settings)
        self._rocket_flight = init_rocket_simulation(self.environment_settings, self.simulation_settings, self.rocket_settings)
        self.is_launched = False

        self.num_timesteps = self._balloon_flights.shape[2]
        self._balloon_status = np.ones((self.balloon_settings["num"], 1))
        self._balloon_states = self._balloon_flights[:, :, self.current_step]
        self._rocket_states = self._rocket_flight.y_sol[:]
        self._rocket_sensors = np.full(12, np.nan)

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1

        if action['launch']:
            self.is_launched = True
        if self.is_launched:
            self._rocket_flight.rocket.roll_control.roll_torque = action['roll']
            self._rocket_flight.rocket.tvc.gimbal_angle_x = action['tvc'][0]
            self._rocket_flight.rocket.tvc.gimbal_angle_y = action['tvc'][1]
            self._rocket_flight.step_simulation()
            self._rocket_sensors[ :3] = self._rocket_flight.sensors[0].measurement   # gyro
            self._rocket_sensors[3:6] = self._rocket_flight.sensors[1].measurement   # accel

        self._balloon_states = self._balloon_flights[:, :, self.current_step]
        self._rocket_states = self._rocket_flight.y_sol[:]
        self._detect_collision()        

        # An episode is done iff reaches max time or end of trajectory
        terminated = (self.current_step >= self.num_timesteps - 1) or (self._rocket_flight._step_state["finished"])
        reward = np.sum(self._balloon_status[:, 0] == 2)
        observation = self._get_obs()
        info = self._get_info()

        if np.remainder(self.current_step,0.1/self.simulation_settings['time_step']) == 0 or terminated:
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _detect_collision(self):
        """Detect collision between balloons and rocket.
        
        If a balloon is released (status=1) and the distance to rocket position
        is less than the balloon radius, set the balloon status to popped (status=2).
        """
        # Calculate distances from each balloon to the rocket
        distances = np.linalg.norm(self._balloon_states[:, :3] - self._rocket_states[:3], axis=1)
        
        # Find balloons that are released and colliding
        released_and_colliding = (distances < self.balloon_settings["radius"]) & (self._balloon_status[:, 0] == 1)
        
        # Update status to popped for collided balloons
        self._balloon_status[released_and_colliding, 0] = 2

    def _render_frame(self):
        if self.render_mode == "vpython":
            if self.render_canvas is None:
                self.render_canvas = canvas(title="Balloon Popping Environment", width=800, height=600, center=vector(0,0,0), background=color.white)
                self.render_balloons = sphere(radius = 1.5, color = color.magenta, make_trail=True)

            self.render_balloons.pos = vector(self._balloon_states[0, 0], self._balloon_states[0, 1], self._balloon_states[0, 2])
            rate(30)
        elif self.render_mode == "matplotlib":
            if self.render_canvas is None:
                self.render_canvas = plt.figure().add_subplot(projection='3d')
                self.render_balloons = self.render_canvas.scatter(self._balloon_states[:, 0], self._balloon_states[:, 1], self._balloon_states[:, 2], c='magenta')
                self.render_rocket = self.render_canvas.plot(self._rocket_states[0], self._rocket_states[1], self._rocket_states[2], 's', color='blue')
                self.render_canvas.set_xlabel('X position (m)')
                self.render_canvas.set_ylabel('Y position (m)')
                self.render_canvas.set_zlabel('Z position (m)')
                self.render_canvas.set_xlim(self._balloon_flights[:, 0,:].min()-10, self._balloon_flights[:, 0,:].max()+10)
                self.render_canvas.set_ylim(self._balloon_flights[:, 1,:].min()-10, self._balloon_flights[:, 1,:].max()+10)
                self.render_canvas.set_zlim(0, self._balloon_flights[:, 2,:].max()+10)

            # Update balloon positions and colors based on status (red if popped)
            colors = ['red' if status == 2 else 'magenta' for status in self._balloon_status[:, 0]]
            self.render_balloons._offsets3d = (self._balloon_states[:, 0], self._balloon_states[:, 1], self._balloon_states[:, 2])
            self.render_balloons.set_facecolors(colors)
            self.render_rocket[0].set_data([self._rocket_states[0]], [self._rocket_states[1]])
            self.render_rocket[0].set_3d_properties([self._rocket_states[2]])
            self.render_canvas.set_title(f"Time: {self.current_step*self.simulation_settings['time_step']} sec\nReward: {np.sum(self._balloon_status[:, 0] == 2)}")
            plt.draw()
            plt.pause(0.1)

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
        latitude=balloon_settings["stochastic"]["latitude_std"],
        longitude=balloon_settings["stochastic"]["longitude_std"],
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
        mass=balloon_settings["stochastic"]["mass_std"],
        volume=balloon_settings["stochastic"]["volume_std"],
        inertia_11=balloon_settings["stochastic"]["inertia_std"],
        inertia_22=balloon_settings["stochastic"]["inertia_std"],
        inertia_33=balloon_settings["stochastic"]["inertia_std"],
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
            "lat0": lambda flight: flight.latitude(0),
            "lon0": lambda flight: flight.longitude(0),
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
    east0, north0, up0 = pm.geodetic2enu(
        monte_carlo_results_["lat0"], monte_carlo_results_["lon0"], environment_settings["elevation"], 
        environment_settings["latitude"], environment_settings["longitude"], environment_settings["elevation"])
    # Broadcast initial ENU offsets to all timesteps for each simulation
    east0, north0, up0 = np.array(east0)[:, None], np.array(north0)[:, None], np.array(up0)[:, None]

    monte_carlo_results = np.stack([
        np.array(monte_carlo_results_["x"]) + east0,
        np.array(monte_carlo_results_["y"]) + north0,
        np.array(monte_carlo_results_["z"]) + up0,
        np.array(monte_carlo_results_["vx"]), 
        np.array(monte_carlo_results_["vy"]), 
        np.array(monte_carlo_results_["vz"]),
    ], axis=1)

    # Debug mode: overwrite with simple test data
    if "debug" in balloon_settings:
        num_balloons = monte_carlo_results.shape[0]
        z_values = 10 + environment_settings["elevation"] + np.arange(num_balloons) * 10  # Spaced 5 m apart
        monte_carlo_results[:, 0, :] = 0  # x = 0
        monte_carlo_results[:, 1, :] = 0  # y = 0  
        monte_carlo_results[:, 2, :] = z_values[:, None]  # z = constant per balloon, 1m apart
        monte_carlo_results[:, 3, :] = 0  # vx = 0
        monte_carlo_results[:, 4, :] = 0  # vy = 0
        monte_carlo_results[:, 5, :] = 0  # vz = 0

    return monte_carlo_results

def init_rocket_simulation(environment_settings, simulation_settings, rocket_settings):
    # Rocket flight simulation initialization
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

    oxidizer_liq = Fluid(name="N2O_l", density=960)
    oxidizer_gas = Fluid(name="N2O_g", density=1.9277)
    tank_shape = CylindricalTank(70 / 1000, 320 / 1000)
    oxidizer_tank = MassFlowRateBasedTank(
        name="oxidizer_tank",
        geometry=tank_shape,
        flux_time=(5),
        initial_liquid_mass=4.3,
        initial_gas_mass=0,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=4.2 / 5,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=oxidizer_liq,
        gas=oxidizer_gas,
    )
    hybrid_motor = HybridMotor(
        thrust_source=rocket_settings["thrust_source"],
        dry_mass=10670 / 1000,
        dry_inertia=(1.668, 1.668, 0.026),
        center_of_dry_mass_position=780 / 1000,
        burn_time=5,
        reshape_thrust_curve=False,
        grain_number=1,
        grain_separation=0,
        grain_outer_radius=43 / 1000,
        grain_initial_inner_radius=22.5 / 1000,
        grain_initial_height=310 / 1000,
        grain_density=920,
        nozzle_radius=0.0141,
        throat_radius=0.00677,
        interpolation_method="linear",
        grains_center_of_mass_position=385 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    hybrid_motor.add_tank(tank=oxidizer_tank, position=934.75 / 1000)

    rocket = Rocket(
        radius=152.4 / 2000,
        mass=14613 / 1000,
        inertia=(24.56, 24.56, 70.074),
        center_of_mass_without_motor=2344 / 1000,
        power_off_drag=rocket_settings["power_off_drag"],
        power_on_drag=rocket_settings["power_on_drag"],
        coordinate_system_orientation="tail_to_nose",
        volume=None,
    )
    rocket.set_rail_buttons(2.808, 1.549)
    rocket.add_motor(hybrid_motor, position=20 / 1000)

    NoseCone = rocket.add_nose(length=0.46, kind="vonKarman", position=3556 / 1000)
    FinSet = rocket.add_trapezoidal_fins(
        n=4,
        span=0.125,
        root_chord=0.247,
        tip_chord=0.045,
        position=0.263,
    )
    Tail = rocket.add_tail(
        top_radius=152.4 / 2000, bottom_radius=0.0496, length=0.254, position=0.254
    )

    gyro_clean = Gyroscope(sampling_rate=rocket_settings["sensor_frequency"])
    accelerometer_clean = Accelerometer(sampling_rate=rocket_settings["sensor_frequency"])
    gnss_clean = GnssReceiver(sampling_rate=rocket_settings["sensor_frequency"])
    rocket.add_sensor(gyro_clean, position=1.5)
    rocket.add_sensor(accelerometer_clean, position=1.5)

    def tvc_controller_function(
        time, sampling_rate, state, state_history, observed_variables, tvc, sensors
    ):
        # log tvc angles
        return (
            time,
            tvc.gimbal_angle_x,
            tvc.gimbal_angle_y,
        )
    tvc, tvc_controller = rocket.add_tvc(
        gimbal_range=rocket_settings["tvc_gimbal_range"],
        sampling_rate=rocket_settings["control_frequency"],
        controller_function=tvc_controller_function,
        return_controller=True,
    )

    def roll_controller_function(
        time, sampling_rate, state, state_history, observed_variables, roll_control, sensors
    ):
        # log roll control torques
        return (
            time,
            roll_control.roll_torque,
        )

    roll_control, roll_controller = rocket.add_roll_control(
        max_roll_torque=rocket_settings["max_roll_torque"],
        sampling_rate=rocket_settings["control_frequency"],
        controller_function=roll_controller_function,
        return_controller=True,
    )

    return Flight(
        rocket=rocket,
        environment=env,
        inclination=90,
        heading=180,
        rail_length=0.1,
        max_time=simulation_settings["max_time"],
        min_time_step=simulation_settings["time_step"]/10,
        time_overshoot=False,
        verbose=True,
    )