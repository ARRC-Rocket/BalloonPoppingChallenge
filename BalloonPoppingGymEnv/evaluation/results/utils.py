import hashlib
import json
import os
import pickle
import urllib.request
from datetime import datetime, timezone

from rocketpy._encoders import RocketPyEncoder


def save_trajectories(trajectories):
    """Save trajectory data as a timestamped JSON list."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_trajectory.json",
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(trajectories, file, indent=2)


def pack_for_submission(eval_cfg, env, scenario_parameters):

    team_name = eval_cfg["team_name"]
    timestamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    # Check the md5 of evaluate.py on local and git main
    url = "https://raw.githubusercontent.com/ARRC-Rocket/BalloonPoppingChallenge/refs/heads/main/BalloonPoppingGymEnv/evaluation/evaluate.py"
    with urllib.request.urlopen(url) as response:
        remote_md5 = hashlib.md5(response.read()).hexdigest()

    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluate.py")
    with open(local, "rb") as f:
        local_md5 = hashlib.md5(f.read()).hexdigest()

    if remote_md5 != local_md5:
        print("Result encryption warning: evaluate.py should not be modified")
        # return

    # Read agent source
    agent_module_path = os.fspath(eval_cfg["agent_module_path"])
    with open(agent_module_path, "r", encoding="utf-8") as f:
        agent_module_file = f.read()

    # Submission payload
    submission = {
        "format_version": 0,
        "team": {
            "name": team_name,
            "secret": eval_cfg["team_secret"],
        },
        "leaderboard_info": {
            "team_name": team_name,
            "timestamp_utc": timestamp,
            "agent_name": eval_cfg["agent_name"],
            "scenario_number": eval_cfg["scenario_number"],
            "final_reward": env._popped_count,
        },
        "balloon_world_data": {
            "scenario_parameters": scenario_parameters,
            "trajectories": env.trajectories,
            "balloon_release_at_step": env._balloon_release_at_step,
            "rocket_flight": json.dumps(env._rocket_flight, cls=RocketPyEncoder),
            "balloon_flights": env._balloon_flights,
        },
        "agent_info": {
            "eval_cfg": eval_cfg,
            "agent_module_file": agent_module_file,
        },
    }

    # Save submission
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{timestamp}_{team_name}_submission.pkl",
    )
    with open(out_path, "wb") as f:
        pickle.dump(submission, f)

    print(f"Submission saved to:\n{out_path}")


def render_trajectory_from_file(file_path):
    """Render trajectory from a saved JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        trajectories = json.load(file)

    # Here you would implement the logic to render the trajectory using your environment's rendering capabilities.
    # This is a placeholder and should be replaced with actual rendering code.
    for step in trajectories:
        rocket_position = step["rocket_states"][:3]  # x, y, z in launch frame
        rocket_velocity = step["rocket_states"][3:6]  # vx, vy, vz in launch frame
        rocket_attitude = step["rocket_states"][6:10]  # quaternion
        rocket_angular_rate = step["rocket_states"][10:13]  # wx, wy, wz in body frame

        balloon_positions = [
            balloon[:3] for balloon in step["balloon_states"]
        ]  # list of x, y, z for each balloon
        balloon_velocities = [
            balloon[3:6] for balloon in step["balloon_states"]
        ]  # list of vx, vy, vz for each balloon
        balloon_status = step["balloon_status"][0]

        print(rocket_position)
        # TODO: render the rocket and balloons using the extracted data
