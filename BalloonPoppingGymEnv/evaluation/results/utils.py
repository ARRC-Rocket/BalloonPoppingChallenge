import hashlib
import http.client
import json
import math
import os
import urllib.request
from datetime import datetime, timezone

import numpy as np
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


def _json_safe(value):
    """Return ``value`` with every non-finite float replaced by ``None``.

    ``json.dump`` writes ``NaN`` and ``Infinity`` for non-finite floats. Python
    reads those back, so the pipeline works either way, but RFC 8259 has no such
    tokens: ``jq`` and a browser ``JSON.parse`` both refuse the file. Since the
    submission now carries a ``.json`` name, it should be a file that any JSON
    reader accepts.

    Rocket states before launch are ``NaN``, so this is ordinary data rather than
    an edge case. ``None`` is also what the leaderboard's replay builder already
    substitutes before anything reaches the viewer, so this only moves that step
    upstream, and the viewer sees no change.

    A float array with nothing to fix is returned untouched so the encoder can
    stream it. Converting it here would turn tens of megabytes of ``float64``
    into individually allocated Python floats for no benefit.
    """
    if isinstance(value, (float, np.floating)):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        if value.dtype.kind != "f":
            return value
        if np.isfinite(value).all():
            return value
        # object dtype, so the None survives tolist()
        return np.where(np.isfinite(value), value, None).tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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

    # The encoder can turn the Flight into JSON, but only as text, so its own
    # non-finite floats would slip past _json_safe below. Round-tripping it here
    # brings them into reach. The flight is a few megabytes, so the extra pass is
    # not worth avoiding.
    rocket_flight = json.loads(
        json.dumps(env._rocket_flight, cls=RocketPyEncoder, allow_pickle=False)
    )

    # Submission payload
    submission = {
        "format_version": 1,
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
            "rocket_flight": rocket_flight,
            "balloon_flights": env._balloon_flights,
        },
        "agent_info": {
            "eval_cfg": eval_cfg,
            "agent_module_file": agent_module_file,
        },
    }
    submission = _json_safe(submission)

    # Save submission as JSON. The upload endpoint is unauthenticated, so the
    # format must not be able to execute code on load the way pickle did (the
    # leaderboard side is issue #7); json.loads cannot. RocketPyEncoder turns the
    # numpy arrays and the Flight object into plain JSON. allow_pickle=False keeps
    # callables out of the payload: with the default the encoder hex-encodes a
    # Function's callable source with dill, which is inert under json.loads but
    # would be executable material for anything that later runs dill.loads on it.
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{timestamp}_{team_name}_submission.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, cls=RocketPyEncoder, allow_pickle=False)

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
