import hashlib
import http.client
import json
import os
import pickle
import re
import tempfile
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


# Path separators, the Windows-reserved punctuation, and control characters. A
# whitelist of ASCII would be shorter, but it would also rewrite every non-ASCII
# team name, and this competition has plenty of those.
_UNSAFE_IN_FILENAME = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]+')

# Filesystem name limits are in bytes, not characters, so the cap has to be too.
# 96 leaves room for the timestamp and the suffix under a 255-byte limit even if
# every character costs four bytes.
_MAX_SLUG_BYTES = 96


def _filename_slug(raw_name, fallback="team"):
    """Reduce a team name to something that cannot leave the results directory.

    ``team_name`` is organizer-assigned rather than attacker-controlled, but it
    reaches the output path verbatim, and a name holding a path separator or
    ``..`` would write outside ``results/``. Only the filename is rewritten: the
    submission payload keeps the name exactly as configured, and that is where
    the leaderboard reads it from, so nothing downstream sees the slug.

    Only genuinely unsafe characters are replaced. A name in Chinese or Japanese
    stays readable, which is the point: the filename is how a competitor tells
    their own submissions apart. Leading dots go so the file cannot hide itself or
    turn into a parent reference, and a trailing dot or space goes because Windows
    will not keep one.

    An empty result falls back rather than raising. By the time this runs the
    simulation is already done, and the same reasoning as the integrity check
    applies: nothing at packing time should cost a competitor a finished run.
    """
    slug = _UNSAFE_IN_FILENAME.sub("_", str(raw_name)).strip(" ._-")
    # Truncating bytes can split a character, so drop the partial tail, then strip
    # again in case the cut exposed a trailing dot.
    slug = slug.encode("utf-8")[:_MAX_SLUG_BYTES].decode("utf-8", "ignore")
    return slug.strip(" ._-") or fallback


def _write_atomically(out_path, write_payload):
    """Write via a temp file in the same directory, then ``os.replace`` it.

    A scenario-1 submission runs to a few hundred megabytes. Writing straight to
    the final path means a disk-full or a killed process leaves a truncated file
    under a name that looks finished, with nothing to distinguish it from a good
    submission. Same-directory ``os.replace`` is atomic on POSIX and Windows, so
    the final path only ever holds a complete file.

    ``write_payload`` receives the open binary handle. The flush and fsync are
    what make the rename meaningful: without them a crash can leave the renamed
    file holding nothing. A disk that fills up surfaces as an error from one of
    the two, which is the case this is here for, so neither is softened.

    One deliberate difference from a plain ``open``: ``mkstemp`` creates the file
    0600 rather than at the umask default, and the mode survives the rename. The
    payload carries the team secret, so keeping it owner-only is the better end
    state on a shared machine.
    """
    handle, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(out_path) or ".", prefix=".partial_", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as file:
            write_payload(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, out_path)
    except BaseException:
        # Including KeyboardInterrupt: a stray .partial_ left behind would be a
        # worse outcome than the truncated file this exists to prevent.
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def pack_for_submission(eval_cfg, env, scenario_parameters):

    team_name = eval_cfg["team_name"]
    packed_at = datetime.now(timezone.utc)
    timestamp = f"{packed_at:%Y%m%dT%H%M%SZ}"

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

    # Save submission. The filename carries milliseconds and a path-safe team
    # name, so two runs in the same second no longer overwrite each other. The
    # payload above keeps the second-resolution timestamp and the configured team
    # name untouched, since those are the copies the leaderboard reads.
    file_timestamp = f"{packed_at:%Y%m%dT%H%M%S}.{packed_at.microsecond // 1000:03d}Z"
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{file_timestamp}_{_filename_slug(team_name)}_submission.pkl",
    )
    _write_atomically(out_path, lambda file: pickle.dump(submission, file))

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
