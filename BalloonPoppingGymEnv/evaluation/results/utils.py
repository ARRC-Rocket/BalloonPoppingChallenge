import json
import os
from datetime import datetime


def save_trajectories(trajectories):
    """Save trajectory data as a timestamped JSON list."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{datetime.now():%y%m%d%H%M%S}_trajectory.json",
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(trajectories, file, indent=2)


def render_trajectory_from_file(file_path):
    """Render trajectory from a saved JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        trajectories = json.load(file)

    # Here you would implement the logic to render the trajectory using your environment's rendering capabilities.
    # This is a placeholder and should be replaced with actual rendering code.
    for step in trajectories:
        rocket_position = step["rocket"][:3]  # x, y, z in launch frame
        rocket_velocity = step["rocket"][3:6]  # vx, vy, vz in launch frame
        rocket_attitude = step["rocket"][6:10]  # quaternion
        rocket_angular_rate = step["rocket"][10:13]  # wx, wy, wz in body frame

        balloon_positions = [
            balloon[:3] for balloon in step["balloons"]
        ]  # list of x, y, z for each balloon
        balloon_velocities = [
            balloon[3:6] for balloon in step["balloons"]
        ]  # list of vx, vy, vz for each balloon

        print(rocket_position)
        # TODO: render the rocket and balloons using the extracted data
