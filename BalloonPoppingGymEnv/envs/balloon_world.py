import gymnasium as gym
from gymnasium import spaces
import numpy as np
from vpython import canvas, color, vector, rate, sphere, arrow
import matplotlib.pyplot as plt
import pymap3d as pm

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
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gyroscope import Gyroscope
from rocketpy.sensors.gnss_receiver import GnssReceiver
from rocketpy.tools import euler313_to_quaternions

class BalloonPoppingEnv(gym.Env):
    metadata = {"render_modes": ["vpython", "matplotlib"]}
    def __init__(self, settings):

        self.scenario_number = settings["scenario_number"]
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
                "launch_inclination_heading": spaces.Box(low=np.array([0, 0]), high=np.array([90, 360]), shape=(2,), dtype=np.float64),
                "tvc":  spaces.Box(low=-self.rocket_settings["control"]["tvc_gimbal_range"]*np.ones(2), high=self.rocket_settings["control"]["tvc_gimbal_range"]*np.ones(2), dtype=np.float64),
                "throttle": spaces.Box(low=self.rocket_settings["control"]["throttle_range"][0], high=self.rocket_settings["control"]["throttle_range"][1], shape=(1,), dtype=np.float64),
                "roll": spaces.Box(low=-self.rocket_settings["control"]["max_roll_torque"], high=self.rocket_settings["control"]["max_roll_torque"], dtype=np.float64),
            }
        )

        # Create environment for balloon flights and rocket simulation
        self.__create_environment()

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
            "rocket_sensors": self._rocket_sensors
        }

    def _get_info(self):
        return {"rocket_states": self._rocket_states}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.__generate_balloon_flights()

        if self.scenario_number == 0: # Scenario 0: hello world with static balloons
            self._balloon_status = np.ones((self.balloon_settings["num"], 1))
            num_balloons = self._balloon_flights.shape[0]
            z_values = 10 + self._rocketpy_env.elevation + np.arange(num_balloons) * 40  # Spaced 40 m apart
            self._balloon_flights[:, [0, 1, 3, 4, 5], :] = 0    # x, y, vx, vy, vz = 0 for static balloons
            self._balloon_flights[:, 2, :] = z_values[:, None]  # z = constant per balloon
        else:
            self._balloon_status = np.zeros((self.balloon_settings["num"], 1))

        self.rocket_launched = False
        self.current_step = 0
        self.num_timesteps = self._balloon_flights.shape[2]
        self._balloon_states = self._balloon_flights[:, :, self.current_step]
        self._rocket_sensors = np.full(12, np.nan)

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def step(self, action):
        previous_balloon_positions = self._balloon_states[:, :3].copy()
        previous_rocket_position = self._rocket_states[:3].copy()
        self.current_step += 1

        #  Update the balloon states
        self._balloon_states = self._balloon_flights[:, :, self.current_step]

        if not self.rocket_launched:
            _rocket_finished = False
            if  action['launch']:   # Init rocket flight with first launch action
                self.rocket_launched = True
                self.__get_init_rocket_states(action['launch_inclination_heading'][0], action['launch_inclination_heading'][1])
                self.initial_solution[0] = self.current_step*self.simulation_settings['time_step']
                self.__init_rocket_simulation()
        else:   # Apply action to step the rocket simulation and get sensor measurements
            self._rocket_flight.rocket.roll_control.roll_torque = action['roll']
            self._rocket_flight.rocket.tvc.gimbal_angle_x = action['tvc'][0]
            self._rocket_flight.rocket.tvc.gimbal_angle_y = action['tvc'][1]
            self._rocket_flight.step_simulation()
            self._rocket_sensors[ :3] = self._rocket_flight.sensors[0].measurement  # gyro
            self._rocket_sensors[3:6] = self._rocket_flight.sensors[1].measurement  # accel
            self._rocket_sensors[6:12] = self._rocket_flight.sensors[2].measurement # gnss
            self._rocket_states = self._rocket_flight.y_sol[:]
            _rocket_finished = (self._rocket_flight._step_state["finished"])
            
            # detect collisions
            self._detect_collision(previous_balloon_positions, previous_rocket_position)

        # An episode is done iff reaches max time or end of trajectory
        terminated = (self.current_step >= self.num_timesteps - 1) or _rocket_finished
        reward = np.sum(self._balloon_status[:, 0] == 2)
        observation = self._get_obs()
        info = self._get_info()

        if np.remainder(self.current_step,0.1/self.simulation_settings['time_step']) == 0 or terminated:
            self._render_frame()

        return observation, reward, terminated, False, info
    
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

        epsilon = 1e-12
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

                non_parallel = regular & (np.abs(denominator) > epsilon)
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

    def _detect_collision(self, previous_balloon_positions, previous_rocket_position):
        """Detect collisions using swept paths over the current timestep."""
        previous_balloon_positions = np.asarray(previous_balloon_positions, dtype=float)
        previous_rocket_position = np.asarray(previous_rocket_position, dtype=float)
        current_balloon_positions = np.asarray(self._balloon_states[:, :3], dtype=float)
        current_rocket_position = np.asarray(self._rocket_states[:3], dtype=float)
        balloon_radius_squared = self.balloon_settings["radius"] ** 2
        released_mask = self._balloon_status[:, 0] == 1
        if not np.any(released_mask):
            return

        distance_squared = self._segment_distance_squared_batch(
            previous_rocket_position,
            current_rocket_position,
            previous_balloon_positions[released_mask],
            current_balloon_positions[released_mask],
        )
        collided_released = distance_squared <= balloon_radius_squared
        released_indices = np.flatnonzero(released_mask)
        self._balloon_status[released_indices[collided_released], 0] = 2

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
            self.render_canvas.set_title(f"Time: {self.current_step*self.simulation_settings['time_step']:.2f} sec\nReward: {np.sum(self._balloon_status[:, 0] == 2)}")
            plt.draw()
            plt.pause(0.001)
        else:
            pass

    def close(self):
        print('closing environment')

    def __create_environment(self):
        self._rocketpy_env = Environment(
            date=self.environment_settings["date"],
            latitude=self.environment_settings["latitude"],
            longitude=self.environment_settings["longitude"],
            elevation=self.environment_settings["elevation"],
            datum="WGS84",
            timezone="UTC",
        )
        if self.environment_settings["use_standard_atmosphere"]:
            self._rocketpy_env.set_atmospheric_model(type="standard_atmosphere")
        else:
            if self.environment_settings["atmosphere_data_path"] == "":
                raise ValueError("Atmosphere data path must be provided if not using standard atmosphere.")
            self._rocketpy_env.set_atmospheric_model(
                type="Ensemble",
                file=self.environment_settings["atmosphere_data_path"],
                dictionary="ECMWF",
            )
    
    def __get_init_rocket_states(self, inclination, heading):
        # Initialize time and state variables
        t_initial = 0
        x_init, y_init, z_init = 0, 0, self.environment_settings["elevation"]
        vx_init, vy_init, vz_init = 0, 0, 0
        w1_init, w2_init, w3_init = 0, 0, 0
        # Initialize attitude
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

    def __generate_balloon_flights(self):

        stochastic_env = StochasticEnvironment(
            environment=self._rocketpy_env,
            latitude=self.balloon_settings["stochastic"]["latitude_std"],
            longitude=self.balloon_settings["stochastic"]["longitude_std"],
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

        cL0 = self.balloon_settings["aero_coefficients"]["cL"]
        cQ0 = self.balloon_settings["aero_coefficients"]["cQ"]
        cD0 = self.balloon_settings["aero_coefficients"]["cD"]
        cDamping = self.balloon_settings["aero_coefficients"]["moment_damping"]
        balloon_aero_model = LinearGenericSurface(
            reference_area=np.pi * self.balloon_settings["radius"] ** 2,
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
            volume= 4 / 3 * np.pi * self.balloon_settings["radius"] ** 3,
            radius=0.05,
            mass=self.balloon_settings["mass"],
            inertia=self.balloon_settings["inertia"],
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
            mass=self.balloon_settings["stochastic"]["mass_std"],
            volume=self.balloon_settings["stochastic"]["volume_std"],
            inertia_11=self.balloon_settings["stochastic"]["inertia_std"],
            inertia_22=self.balloon_settings["stochastic"]["inertia_std"],
            inertia_33=self.balloon_settings["stochastic"]["inertia_std"],
            center_of_mass_without_motor=0,
        )
        stochastic_balloon.add_motor(SM, position=0)
        stochastic_balloon.add_linear_generic_surface(balloon_aero_model)

        flight = Flight(
            rocket=Balloon,
            environment=self._rocketpy_env,
            inclination=90,
            heading=180,
            rail_length=0.1,
            max_time=self.simulation_settings["max_time"],
            # max_time_step=0.01,
            # min_time_step=0.01,
            verbose=True,
        )
        stochastic_flight = StochasticFlight(
            flight=flight,
            inclination=5,
            heading=90,
        )

        monte_carlo_sim = MonteCarlo(
            filename="./BalloonPoppingGymEnv/envs/data/balloon_sim",
            environment=stochastic_env,
            rocket=stochastic_balloon,
            flight=stochastic_flight,
            export_list=["t_final"],
            data_collector={
                "x": lambda flight: flight.x(np.arange(0, self.simulation_settings["max_time"], self.simulation_settings["time_step"])),
                "y": lambda flight: flight.y(np.arange(0, self.simulation_settings["max_time"], self.simulation_settings["time_step"])),
                "z": lambda flight: flight.z(np.arange(0, self.simulation_settings["max_time"], self.simulation_settings["time_step"])),
                "vx": lambda flight: flight.vx(np.arange(0, self.simulation_settings["max_time"], self.simulation_settings["time_step"])),
                "vy": lambda flight: flight.vy(np.arange(0, self.simulation_settings["max_time"], self.simulation_settings["time_step"])),
                "vz": lambda flight: flight.vz(np.arange(0, self.simulation_settings["max_time"], self.simulation_settings["time_step"])),
                "lat0": lambda flight: flight.latitude(0),
                "lon0": lambda flight: flight.longitude(0),
            },
        )

        monte_carlo_results_ = monte_carlo_sim.simulate(
            number_of_simulations=self.balloon_settings["num"],
            append=False,
            include_function_data=False,
            parallel=False,
            n_workers=4,
        )

        """Convert Monte Carlo dict to [balloon][state][timestep]."""
        east0, north0, up0 = pm.geodetic2enu(
            monte_carlo_results_["lat0"], monte_carlo_results_["lon0"], self._rocketpy_env.elevation, 
            self._rocketpy_env.latitude, self._rocketpy_env.longitude, self._rocketpy_env.elevation)
        # Broadcast initial ENU offsets to all timesteps for each simulation
        east0, north0, up0 = np.array(east0)[:, None], np.array(north0)[:, None], np.array(up0)[:, None]

        self._balloon_flights = np.stack([
            np.array(monte_carlo_results_["x"]) + east0,
            np.array(monte_carlo_results_["y"]) + north0,
            np.array(monte_carlo_results_["z"]) + up0,
            np.array(monte_carlo_results_["vx"]), 
            np.array(monte_carlo_results_["vy"]), 
            np.array(monte_carlo_results_["vz"]),
        ], axis=1)

    def __init_rocket_simulation(self):
        # Rocket flight simulation initialization

        # Create tank fluids from settings
        tank_cfg = self.rocket_settings["tank"]
        oxidizer_gas = Fluid(name=tank_cfg["gas"],density=tank_cfg["gas_density"])
        oxidizer_liq = Fluid(name=tank_cfg["liquid"], density=tank_cfg["liquid_density"])
        
        # Create tank from settings
        tank_shape = CylindricalTank(radius=tank_cfg["radius"],height=tank_cfg["height"])
        oxidizer_tank = MassFlowRateBasedTank(
            name="oxidizer_tank",
            geometry=tank_shape,
            flux_time=tank_cfg["flux_time"],
            initial_liquid_mass=tank_cfg["initial_liquid_mass"],
            initial_gas_mass=tank_cfg["initial_gas_mass"],
            liquid_mass_flow_rate_in=0,
            liquid_mass_flow_rate_out=tank_cfg["liquid_mass_flow_rate_out"],
            gas_mass_flow_rate_in=0,
            gas_mass_flow_rate_out=0,
            liquid=oxidizer_liq,
            gas=oxidizer_gas,
            temperature=tank_cfg["temperature"],
            pressure=tank_cfg["pressure"],
        )
        
        # Create motor from settings
        motor_cfg = self.rocket_settings["motor"]
        hybrid_motor = HybridMotor(
            thrust_source=motor_cfg["thrust_source"],
            dry_mass=motor_cfg["dry_mass"],
            dry_inertia=tuple(motor_cfg["dry_inertia"]),
            center_of_dry_mass_position=motor_cfg["center_of_dry_mass_position"],
            burn_time=motor_cfg["burn_time"],
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
        tank_position = -tank_cfg["height"] / 2 - motor_cfg["motor_position"]
        hybrid_motor.add_tank(tank=oxidizer_tank, position=tank_position)

        # Create rocket body from settings
        rocket_cfg = self.rocket_settings["rocket_body"]
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
        
        # Add motor from settings
        rocket.add_motor(hybrid_motor, position=motor_cfg["motor_position"])
        
        # Add nose from settings
        nose_cfg = self.rocket_settings["nose"]
        rocket.add_nose(
            length=nose_cfg["length"],
            kind=nose_cfg["kind"],
            position=nose_cfg["position"]
        )
        
        # Add fins from settings
        fins_cfg = self.rocket_settings["fins"]
        if fins_cfg["useFins"]:
            rocket.add_trapezoidal_fins(
                n=fins_cfg["n"],
                span=fins_cfg["span"],
                root_chord=fins_cfg["root_chord"],
                tip_chord=fins_cfg["tip_chord"],
                position=fins_cfg["position"],
            )

        # Add sensors from settings
        sensors_cfg = self.rocket_settings["sensors"]
        gyro = Gyroscope(
            sampling_rate=sensors_cfg["sampling_rate"],
            noise_density=sensors_cfg["gyro_noise_density"],
            random_walk_density=sensors_cfg["gyro_random_walk_density"],
            constant_bias=sensors_cfg["gyro_constant_bias"],
            )
        accelerometer = Accelerometer(
            sampling_rate=sensors_cfg["sampling_rate"],
            noise_density=sensors_cfg["accelerometer_noise_density"],
            random_walk_density=sensors_cfg["accelerometer_random_walk_density"],
            constant_bias=sensors_cfg["accelerometer_constant_bias"],
            )
        gnss = GnssReceiver(
            sampling_rate=sensors_cfg["sampling_rate"],
            position_accuracy=sensors_cfg["gnss_position_accuracy"],
            altitude_accuracy=sensors_cfg["gnss_altitude_accuracy"],
            velocity_accuracy=sensors_cfg["gnss_velocity_accuracy"],
            )
        rocket.add_sensor(gyro, position=sensors_cfg["gyro_position"])
        rocket.add_sensor(accelerometer, position=sensors_cfg["accelerometer_position"])
        rocket.add_sensor(gnss, position=sensors_cfg["gnss_position"])
        # rocket.draw()

        # Add control systems from settings
        control_cfg = self.rocket_settings["control"]

        def tvc_controller_function(
            time, sampling_rate, state, state_history, observed_variables, tvc, sensors
        ):
            # log tvc angles
            return (
                time,
                tvc.gimbal_angle_x,
                tvc.gimbal_angle_y,
            )
        rocket.add_tvc(
            gimbal_range=control_cfg["tvc_gimbal_range"],
            sampling_rate=control_cfg["control_frequency"],
            controller_function=tvc_controller_function,
            return_controller=False,
        )

        def roll_controller_function(
            time, sampling_rate, state, state_history, observed_variables, roll_control, sensors
        ):
            # log roll control torques
            return (
                time,
                roll_control.roll_torque,
            )

        rocket.add_roll_control(
            max_roll_torque=control_cfg["max_roll_torque"],
            sampling_rate=control_cfg["control_frequency"],
            controller_function=roll_controller_function,
            return_controller=False,
        )

        self._rocket_flight = Flight(
            rocket=rocket,
            environment=self._rocketpy_env,
            rail_length=0.01,    # No rail since we directly set initial conditions to simulate launch
            initial_solution = self.initial_solution,
            max_time=self.simulation_settings["max_time"],
            min_time_step=self.simulation_settings["time_step"]/10,
            time_overshoot=False,
            verbose=True,
        )