import numpy as np

from BalloonPoppingGymEnv.agents.example_agent import AttitudeRateControlAgent

scenario_number = 0
agent_name = "no_action_agent"
agent_kwargs = {"launch_time": 1.0}

def run_for_development():
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

    # Load scenario parameters
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)

    # Create environment with scenario parameters
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)

    # Instantiate agent with given parameters and any additional user kwargs
    agent = AttitudeRateControlAgent(given_parameters, **agent_kwargs)

    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = False

    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)
        print(f"simulation_time: {observation['simulation_time']:.2f} sec, reward: {reward:.2f}", end='\r')

    print(f"Scenario {scenario_number} evaluation completed with agent '{agent_name}'.")
    print(f"Final reward: {reward}")

def run_for_evaluation():
    from BalloonPoppingGymEnv.evaluation.evaluate import evaluate_scenario

    # Load agent class dynamically from specified module path.
    evaluate_scenario(
        AttitudeRateControlAgent,
        agent_kwargs=agent_kwargs,
        agent_name=agent_name,
        scenario_number=scenario_number,
        render_mode='matplotlib',
    )

if __name__ == "__main__":
    # Use this function for development and debugging purposes.
    run_for_development()

    # Use this function for evaluation purposes.
    # run_for_evaluation()