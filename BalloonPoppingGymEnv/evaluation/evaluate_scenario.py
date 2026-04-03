import importlib.util
import sys

import yaml

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


def _extract_nested_parameters(scenario_parameters, given_parameters_spec):
    """Extract subset of parameters based on specification.

    Parameters
    ----------
    scenario_parameters : dict
        Full scenario parameters dictionary
    given_parameters_spec : dict
        Specification of which keys to extract from scenario_parameters

    Returns
    -------
    dict
        Filtered parameters containing only specified keys
    """
    given_parameters = {}

    for section, keys in given_parameters_spec.items():
        if isinstance(keys, list):
            given_parameters[section] = {
                key: scenario_parameters[section][key]
                for key in keys
                if key in scenario_parameters[section]
            }
        elif isinstance(keys, dict):
            given_parameters[section] = {}
            for subsection, sub_keys in keys.items():
                given_parameters[section][subsection] = {
                    key: scenario_parameters[section][subsection][key]
                    for key in sub_keys
                    if key in scenario_parameters[section][subsection]
                }

    return given_parameters


def _load_agent(agent_module_path, agent_cls_name, given_parameters, agent_kwargs):
    """Load agent class dynamically from specified module path.

    Parameters
    ----------
    agent_module_path : str
        Path to the agent module file
    agent_cls_name : str
        Name of the agent class to instantiate
    given_parameters : dict
        Parameters to pass to agent initialization
    agent_kwargs : dict
        Additional keyword arguments for agent initialization

    Returns
    -------
    object
        Instantiated agent object
    """
    spec = importlib.util.spec_from_file_location("agent_module", agent_module_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    AgentClass = getattr(agent_module, agent_cls_name)
    return AgentClass(given_parameters, **agent_kwargs)


def evaluate_scenario(eval_cfg_path):
    """Evaluate a scenario with a given agent configuration.

    Parameters
    ----------
    eval_cfg_path : str
        Path to evaluation configuration YAML file
    """
    with open(eval_cfg_path, "r") as file:
        eval_cfg = yaml.safe_load(file)
    file.close()

    scenario_number = eval_cfg["scenario_number"]
    render_mode = eval_cfg["render_mode"]
    agent_module_path = eval_cfg["agent_module_path"]
    agent_cls_name = eval_cfg["agent_cls"]
    agent_kwargs = eval_cfg["agent_kwargs"]

    with open(
        f"./BalloonPoppingGymEnv/envs/scenario_parameters/scenario_{scenario_number}_parameters.yaml",
        "r",
    ) as file:
        scenario_parameters = yaml.safe_load(file)
    file.close()

    with open(
        f"./BalloonPoppingGymEnv/envs/scenario_parameters/scenario_{scenario_number}_given_parameters.yaml",
        "r",
    ) as file:
        given_parameters_spec = yaml.safe_load(file)
    file.close()

    given_parameters = _extract_nested_parameters(
        scenario_parameters, given_parameters_spec
    )

    # Create environment with scenario parameters
    env = BalloonPoppingEnv(render_mode=render_mode, parameters=scenario_parameters)

    # Load agent class dynamically from specified module path.
    agent = _load_agent(
        agent_module_path, agent_cls_name, given_parameters, agent_kwargs
    )

    observation, info = env.reset()
    terminated = False

    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)

    print(
        f"Scenario {scenario_number} evaluation completed with agent '{eval_cfg['agent_name']}'."
    )
    print(f"Final reward: {reward}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Configuration file path is required. "
            "Usage: python evaluate_scenario.py <path_to_eval_config.yaml>"
        )
    eval_config_path = sys.argv[1]
    evaluate_scenario(eval_config_path)
