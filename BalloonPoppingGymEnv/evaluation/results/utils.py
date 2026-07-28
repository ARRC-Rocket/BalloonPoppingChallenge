import hashlib
import http.client
import json
import os
import pickle
import urllib.request
from datetime import datetime, timezone

from rocketpy._encoders import RocketPyEncoder

# The evaluate.py integrity check is advisory, so it gets a short budget and is
# never allowed to cost a competitor the submission they just simulated.
INTEGRITY_CHECK_TIMEOUT = 10


def save_trajectories(trajectories):
    """Save trajectory data as a timestamped JSON list."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_trajectory.json",
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(trajectories, file, indent=2)


def _normalized_md5(raw_bytes):
    """MD5 of ``raw_bytes`` after dropping a UTF-8 BOM and normalizing line endings.

    ``pack_for_submission`` compares the local ``evaluate.py`` against the copy on
    main. A Windows checkout can add a UTF-8 BOM or CRLF/CR endings that would flag
    an otherwise-unmodified file (issue #35), so both are normalized away first.

    This works on raw bytes and only folds the endings Python treats as a physical
    line break: CRLF, CR, LF. Decoding to text and using ``str.splitlines()``
    instead would be unsafe twice over. ``splitlines()`` also breaks on form feed,
    NEL, U+2028 and friends, which the tokenizer does not, so swapping a single LF
    for a form feed changes the program yet keeps the hash. And decoding as UTF-8
    raises on a valid non-UTF-8 source (one with a ``coding:`` declaration), turning
    the integrity warning into a crash. A real source edit still changes the hash.
    """
    normalized = raw_bytes.removeprefix(b"\xef\xbb\xbf")
    normalized = normalized.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    normalized = normalized.removesuffix(b"\n")
    return hashlib.md5(normalized).hexdigest()


def pack_for_submission(eval_cfg, env, scenario_parameters):

    team_name = eval_cfg["team_name"]
    timestamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    # Compare evaluate.py against the official copy on main (see _normalized_md5).
    # This runs after the whole simulation, so a network problem must not cost the
    # competitor their submission: the check fails open and packing continues.
    url = "https://raw.githubusercontent.com/ARRC-Rocket/BalloonPoppingChallenge/refs/heads/main/BalloonPoppingGymEnv/evaluation/evaluate.py"

    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluate.py")
    with open(local, "rb") as f:
        local_md5 = _normalized_md5(f.read())

    try:
        with urllib.request.urlopen(url, timeout=INTEGRITY_CHECK_TIMEOUT) as response:
            remote_md5 = _normalized_md5(response.read())
    except (OSError, http.client.HTTPException) as exc:
        # URLError, HTTPError, socket timeouts and DNS failures are all OSError.
        print(f"Could not check evaluate.py against the official copy: {exc}")
    else:
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
        # env._balloon_flights is deliberately not here. It is 71% of a
        # scenario-1 submission and carries nothing the rest of the payload does
        # not: step() reads it as _balloon_flights[:, :, current_step] and records
        # the result, so trajectories[k]["balloon_states"] is bit-for-bit
        # _balloon_flights[:, :, k + 1] for the whole simulated window. What the
        # array adds is the frame before the first step and the tail after the
        # episode ended, and neither takes part in scoring or replay.
        #
        # Those two are not archived any more, rather than regenerable. Rerunning
        # the scenario needs the same atmosphere NetCDF bytes, the same
        # ActiveRocketPy commit and the same solver behaviour, and the golden
        # master is downsampled and tolerant, so it says the physics has not
        # drifted, not that a dropped frame comes back byte for byte.
        #
        # format_version stays 0: this is a producer that has stopped emitting an
        # optional key rather than a new schema, and the leaderboard reads
        # trajectories. A third-party tool that indexes world["balloon_flights"]
        # gets a KeyError, so it is called out in the changelog. See issue #57.
        "balloon_world_data": {
            "scenario_parameters": scenario_parameters,
            "trajectories": env.trajectories,
            "balloon_release_at_step": env._balloon_release_at_step,
            "rocket_flight": json.dumps(env._rocket_flight, cls=RocketPyEncoder),
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
        # rocket_velocity = step["rocket_states"][3:6]  # vx, vy, vz in launch frame
        # rocket_attitude = step["rocket_states"][6:10]  # quaternion
        # rocket_angular_rate = step["rocket_states"][10:13]  # wx, wy, wz in body frame

        # balloon_positions = [
        #     balloon[:3] for balloon in step["balloon_states"]
        # ]  # list of x, y, z for each balloon
        # balloon_velocities = [
        #     balloon[3:6] for balloon in step["balloon_states"]
        # ]  # list of vx, vy, vz for each balloon
        # balloon_status = step["balloon_status"][0]

        print(rocket_position)
        # TODO: render the rocket and balloons using the extracted data
