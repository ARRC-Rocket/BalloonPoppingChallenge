import matplotlib.pyplot as plt
import numpy as np

from BalloonPoppingGymEnv.agents.example_agents import NavigationAgent
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

scenario_number = 0

agent_kwargs = {"launch_time": 1.0, "inclination": 90, "heading": 0}

# Load scenario parameters
scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)

# Create environment with scenario parameters turn off rendering to make own plots
env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)

# Instantiate agent with given parameters and any additional user kwargs
agent = NavigationAgent(given_parameters, **agent_kwargs)

# use seed=None to randomize environment
observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
terminated = False

attitude_gt = np.full((4, 1), np.nan)
attitude_est = np.full((4, 1), np.nan)
velocity_gt = np.full((3, 1), np.nan)
velocity_est = np.full((3, 1), np.nan)
time = np.full(1, np.nan)

while not terminated:
    action = agent.get_action(observation)
    observation, reward, terminated, _, info = env.step(action)

    time = np.append(time, observation["simulation_time"])
    # ground truth velocity
    velocity_gt = np.append(velocity_gt, info["rocket_states"][3:6].reshape(-1, 1), axis=1)
    # estimated velocity
    velocity_est = np.append(velocity_est, agent.states[3:6].reshape(-1, 1), axis=1)
    # ground truth attitude
    attitude_gt = np.append(attitude_gt, info["rocket_states"][6:10].reshape(-1, 1), axis=1)
    # estimated attitude
    attitude_est = np.append(attitude_est, agent.states[6:10].reshape(-1, 1), axis=1)
    

    print(f"simulation_time: {observation['simulation_time']:.2f} sec, reward: {reward:.2f}", end='\r')

plt.subplot(2, 1, 1)
plt.plot(time, attitude_gt[0], 'r-', label='e0_attitude_gt')
plt.plot(time, attitude_gt[1], 'g-', label='e1_attitude_gt')
plt.plot(time, attitude_gt[2], 'b-', label='e2_attitude_gt')
plt.plot(time, attitude_gt[3], 'm-', label='e3_attitude_gt')
plt.plot(time, attitude_est[0], 'r--', label='e0_attitude_est')
plt.plot(time, attitude_est[1], 'g--', label='e1_attitude_est')
plt.plot(time, attitude_est[2], 'b--', label='e2_attitude_est')
plt.plot(time, attitude_est[3], 'm--', label='e3_attitude_est')
plt.xlabel('Time (s)')
plt.ylabel('Attitude (quaternion)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, velocity_gt[0], 'r-', label='x_velocity_gt')
plt.plot(time, velocity_gt[1], 'g-', label='y_velocity_gt')
plt.plot(time, velocity_gt[2], 'b-', label='z_velocity_gt')
plt.plot(time, velocity_est[0], 'r--', label='x_velocity_est')
plt.plot(time, velocity_est[1], 'g--', label='y_velocity_est')
plt.plot(time, velocity_est[2], 'b--', label='z_velocity_est')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.tight_layout()
plt.show()

# env._rocket_flight.all_info() # Uncomment to print all info from RocketPy
