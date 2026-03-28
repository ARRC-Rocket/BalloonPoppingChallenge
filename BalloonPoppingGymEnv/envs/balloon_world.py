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

class BalloonPoppingEnv(gym.Env):
    metadata = {"render_modes": ["vpython", "matplotlib"]}
    def __init__(self, settings):

        self.environment_settings = settings["environment"]
        self.simulation_settings = settings["simulation"]
        self.balloon_settings = settings["balloon"]
        self.rocket_settings = settings["rocket"]

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
                "roll": spaces.Box(low=-self.rocket_settings["max_roll_torque"]*np.ones(2), high=self.rocket_settings["max_roll_torque"]*np.ones(2), dtype=np.float64),
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

        self._balloon_flights = generate_balloon_flights(self.environment_settings, self.simulation_settings, self.balloon_settings)
        self._rocket_flight = init_rocket_simulation(self.environment_settings, self.simulation_settings, self.rocket_settings)

        self.num_timesteps = self._balloon_flights.shape[2]
        self._balloon_states = self._balloon_flights[:, :, self.current_step]
        self._rocket_states = np.array(np.zeros(12))

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1

        self._balloon_states = self._balloon_flights[:, :, self.current_step]
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
                self.canvas.set_xlim(self._balloon_flights[:, 0,:].min(), self._balloon_flights[:, 0,:].max())
                self.canvas.set_ylim(self._balloon_flights[:, 1,:].min(), self._balloon_flights[:, 1,:].max())
                self.canvas.set_zlim(0, self._balloon_flights[:, 2,:].max())

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
    rocket.add_sensor(gyro_clean, position=1.5)
    rocket.add_sensor(accelerometer_clean, position=1.5)

    def tvc_controller_function(
        time, sampling_rate, state, state_history, observed_variables, tvc, sensors
    ):
        # state = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

        # print(time)

        tvc.gimbal_angle_x = 0
        tvc.gimbal_angle_y = 0
        # Return variables of interest to be saved in the observed_variables list
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
        # Separate sensor data by type
        gyro_data = None    # rad/s
        accel_data = None   # m/s^2
        
        for sensor in sensors:
            if isinstance(sensor, Gyroscope):
                sensor_time, gx, gy, gz = zip(*sensor.measured_data)
                gyro_data = {
                    'time': np.array(sensor_time),
                    'x': np.array(gx),
                    'y': np.array(gy),
                    'z': np.array(gz)
                }
            elif isinstance(sensor, Accelerometer):
                sensor_time, ax, ay, az = zip(*sensor.measured_data)
                accel_data = {
                    'time': np.array(sensor_time),
                    'x': np.array(ax),
                    'y': np.array(ay),
                    'z': np.array(az)
                }

        KP = 50
        KI = 1
        KD = 0.0

        # Roll rate target: 0.5 Hz sinusoidal, ±10 deg/s
        target = np.deg2rad(10 * np.sin(2 * np.pi * 0.5 * time)) # rad/s

        if gyro_data is None:
            roll_control.roll_torque = 0
            return (time, roll_control.roll_torque,)
        
        roll_rate_errors = target - gyro_data['z']
        roll_rate_error_integral = np.sum(roll_rate_errors) / sampling_rate
        roll_rate_error_derivative = (roll_rate_errors[-1] - roll_rate_errors[-2]) * sampling_rate if len(roll_rate_errors) > 1 else 0
        
        roll_control.roll_torque = KP * roll_rate_errors[-1] + KI * roll_rate_error_integral + KD * roll_rate_error_derivative
        
        # Return variables of interest to be saved in the observed_variables list
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
        time_overshoot=False,
        verbose=False,
    )
def step_rocket_simulation(flight):
    # Placeholder for future rocket simulation logic
    pass