import copy
import importlib.util
import logging
import os
import sys

import yaml

from BalloonPoppingGymEnv.console_logging import configure_console_logging
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.results.utils import save_trajectories

# Named rather than __name__: running this file as a script makes __name__
# "__main__", which puts its records outside the package logger the console
# setup below attaches to, and shows up in the output as "__main__".
logger = logging.getLogger("BalloonPoppingGymEnv.evaluation.evaluate")


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

    Notes
    -----
    Deep copied, because this is what the agent's constructor is handed and the
    agent is a competitor's code. The dict comprehension builds new dicts, so
    replacing a value in the result did not reach the environment, but any list
    value was the same object the environment reads. Measured: four of them were
    shared, and writing to one of those reached the physics.

    ``given["rocket"]["control"]["throttle_range"][1] = 99.0`` made the
    environment build the throttle actuator with a range of ``[0.0, 99.0]``, and
    ``given["rocket"]["rocket_body"]["inertia"][0]`` changed the rocket's moment
    of inertia. Both are read when the flight is constructed on the launch
    action, which is after the agent has already run.
    """
    given_parameters = {}

    for section, keys in given_parameters_spec.items():
        if isinstance(keys, list):
            given_parameters[section] = {
                key: copy.deepcopy(scenario_parameters[section][key])
                for key in keys
                if key in scenario_parameters[section]
            }
        elif isinstance(keys, dict):
            given_parameters[section] = {}
            for subsection, sub_keys in keys.items():
                given_parameters[section][subsection] = {
                    key: copy.deepcopy(scenario_parameters[section][subsection][key])
                    for key in sub_keys
                    if key in scenario_parameters[section][subsection]
                }

    return given_parameters


def _load_agent_class(agent_module_path, agent_cls_name):
    """Load agent class dynamically from specified module path.

    Parameters
    ----------
    agent_module_path : str
        Path to the agent module file
    agent_cls_name : str
        Name of the agent class to instantiate

    Returns
    -------
    type
        Agent class
    """
    spec = importlib.util.spec_from_file_location("agent_module", agent_module_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    AgentClass = getattr(agent_module, agent_cls_name)
    return AgentClass


def load_scenario_parameters(scenario_number):
    """Load scenario parameters from YAML files.

    Parameters
    ----------
    scenario_number : int
        Scenario number to load parameters for

    Returns
    -------
    dict
        Full scenario parameters dictionary
    dict
        Given parameters dictionary extracted based on specification
    """
    parameter_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../envs/scenario_parameters"
    )

    scenario_params_path = os.path.join(
        parameter_dir, f"scenario_{scenario_number}_parameters.yaml"
    )
    with open(scenario_params_path, "r", encoding="utf-8-sig") as file:
        scenario_parameters = yaml.safe_load(file)

    given_params_path = os.path.join(
        parameter_dir, f"scenario_{scenario_number}_given_parameters.yaml"
    )
    with open(given_params_path, "r", encoding="utf-8-sig") as file:
        given_parameters_spec = yaml.safe_load(file)

    given_parameters = _extract_nested_parameters(
        scenario_parameters, given_parameters_spec
    )

    return scenario_parameters, given_parameters


def evaluate_scenario(
    agent_class,
    agent_kwargs=None,
    agent_name="default_agent_name",
    scenario_number=0,
    render_mode=None,
):
    """Evaluate a scenario with given configuration and agent.

    Parameters
    ----------
    agent_class : type
        Agent class to use for evaluation
    agent_kwargs : dict or None
        Additional keyword arguments for agent initialization
    agent_name : str
        Name of the agent for logging purposes
    scenario_number : int
        Scenario number to evaluate
    render_mode : str or None
        Rendering mode for the environment ('matplotlib') or None
    """

    # Load scenario parameters
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)

    # Create environment with scenario parameters
    env = BalloonPoppingEnv(render_mode=render_mode, parameters=scenario_parameters)

    # Instantiate agent with given parameters and any additional user kwargs
    agent = agent_class(given_parameters, **(agent_kwargs or {}))

    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

    save_trajectories(trajectories=env.trajectories)
    logger.info(
        "Scenario %s evaluation completed with agent '%s'.",
        scenario_number,
        agent_name,
    )
    logger.info("Total reward: %s", info["popped_count"])

    return env, agent, scenario_parameters


def main(argv=None):
    """The command line entry point, as a function so it can be driven.

    Split out of the ``__main__`` block so a test can exercise the console
    configuration and the exact output without launching a subprocess that runs
    a full simulation, a renderer and the submission path just to see a
    formatter.
    """
    configure_console_logging()

    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        raise ValueError(
            "Configuration file path is required. "
            "Usage: python evaluate.py <path_to_eval_config.yaml>"
        )

    eval_cfg_path = arguments[0]
    with open(eval_cfg_path, "r", encoding="utf-8-sig") as file:
        eval_cfg = yaml.safe_load(file)

    # Load agent class dynamically from specified module path.
    agent_class = _load_agent_class(
        eval_cfg["agent_module_path"], eval_cfg["agent_class_name"]
    )
    env, agent, scenario_parameters = evaluate_scenario(
        agent_class,
        agent_kwargs=eval_cfg["agent_kwargs"],
        agent_name=eval_cfg["agent_name"],
        scenario_number=eval_cfg["scenario_number"],
        render_mode=eval_cfg["render_mode"],
    )
    if eval_cfg["leaderboard_submission"]:
        from BalloonPoppingGymEnv.evaluation.results.utils import pack_for_submission

        pack_for_submission(
            eval_cfg=eval_cfg, env=env, scenario_parameters=scenario_parameters
        )
    return env, agent, scenario_parameters


if __name__ == "__main__":
    main()
