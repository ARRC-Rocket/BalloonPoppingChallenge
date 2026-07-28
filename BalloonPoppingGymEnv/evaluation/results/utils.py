import hashlib
import http.client
import json
import os
import pickle
import urllib.request
from datetime import datetime, timezone

from rocketpy._encoders import RocketPyEncoder

# The evaluate.py integrity check is advisory and must never cost a competitor
# the submission they just simulated. The socket timeout and the byte cap bound
# each read and the memory it can take, not the total time; what protects the
# run is that the check happens after the file is written.
INTEGRITY_CHECK_URL = "https://raw.githubusercontent.com/ARRC-Rocket/BalloonPoppingChallenge/refs/heads/main/BalloonPoppingGymEnv/evaluation/evaluate.py"
INTEGRITY_CHECK_TIMEOUT = 10
# evaluate.py is a few kB; this leaves plenty of room without being unbounded.
INTEGRITY_CHECK_MAX_BYTES = 1_000_000

# A mismatch is a fact about two files, not a finding about the competitor. The
# old wording ("evaluate.py should not be modified") reads as an accusation, and
# it fires for a reason nobody controls: the reference is whatever sits on main,
# so anyone whose checkout is not current main sees it having edited nothing. It
# is a constant so the tests can key on it instead of on a phrase (see #35).
#
# "A different revision", not "ahead of main": #35 covers both directions. An
# untouched older release starts reporting this the moment main moves on, and
# telling that competitor only about being ahead invites them to conclude the
# file really was edited.
INTEGRITY_MISMATCH_MESSAGE = (
    "evaluate.py does not match the official copy on main. Your submission was "
    "still saved. A checkout from any revision other than current main reports "
    "this too, so check whether you edited the file before treating it as a "
    "problem."
)


def save_trajectories(trajectories):
    """Save trajectory data as a timestamped JSON list."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_trajectory.json",
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(trajectories, file, indent=2)


def _normalized_digest(raw_bytes):
    """SHA-256 of ``raw_bytes``, less a UTF-8 BOM and with line endings normalized.

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
    return hashlib.sha256(normalized).hexdigest()


def _check_evaluate_integrity():
    """Warn when the local evaluate.py differs from the official copy on main.

    Advisory only, and it runs after the whole scenario has been simulated, so
    every expected failure has to leave packing alone: an unreadable local file,
    any network problem, or a reference copy far larger than the real one.

    ``timeout`` bounds each blocking socket operation rather than the whole
    request, so a peer dripping bytes slowly can keep the transfer alive well
    past it. The byte cap bounds memory, not elapsed time; what keeps a slow
    peer from costing anyone their run is that this is called after the
    submission has already been written.
    """
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluate.py")
    try:
        with open(local, "rb") as source:
            # Same cap as the remote side: an unexpectedly large local file
            # should report that the check could not run, not exhaust memory.
            local_raw = source.read(INTEGRITY_CHECK_MAX_BYTES + 1)
        if len(local_raw) > INTEGRITY_CHECK_MAX_BYTES:
            print("Could not check evaluate.py: the local copy is unexpectedly large")
            return
        local_digest = _normalized_digest(local_raw)

        with urllib.request.urlopen(
            INTEGRITY_CHECK_URL, timeout=INTEGRITY_CHECK_TIMEOUT
        ) as response:
            # urlopen treats the whole 2xx range as success, so nothing above
            # raises for a 203 an intermediary rewrote, a 204 with no body, or a
            # 206 carrying one slice of the file. Only 200 is the representation
            # of the target resource, and the completeness check below cannot
            # stand in for this: a well-formed 206 declares the length of its
            # own partial body, matches it, and would be hashed as if it were
            # the whole file.
            status = getattr(response, "status", None)
            if status != 200:
                print(f"Could not check evaluate.py: unexpected HTTP status {status}")
                return
            declared = response.headers.get("Content-Length")
            # One byte over the cap, so an oversized body is detectable.
            raw = response.read(INTEGRITY_CHECK_MAX_BYTES + 1)

        if len(raw) > INTEGRITY_CHECK_MAX_BYTES:
            print(
                "Could not check evaluate.py: the reference copy is unexpectedly large"
            )
            return

        # A peer that announces a length and then closes early hands back a short
        # body without raising, and hashing that would accuse the competitor of
        # editing a file they never touched. A length that will not parse, or
        # parses negative, says the framing itself cannot be trusted, which is
        # the same conclusion.
        if declared is not None:
            try:
                expected = int(declared)
            except ValueError:
                print("Could not check evaluate.py: unusable Content-Length")
                return
            if expected < 0:
                print("Could not check evaluate.py: unusable Content-Length")
                return
            if len(raw) < expected:
                print(
                    "Could not check evaluate.py: the reference copy arrived incomplete"
                )
                return

        # Inside the try with the local digest, so a build that refuses this hash
        # reports an unavailable check from either side rather than only one.
        remote_digest = _normalized_digest(raw)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # URLError, HTTPError, socket timeouts and DNS failures are all OSError.
        # ValueError covers a hash the interpreter refuses to provide, which is
        # how a FIPS-restricted build reports a blocked digest.
        print(f"Could not check evaluate.py against the official copy: {exc}")
        return

    if remote_digest != local_digest:
        print(INTEGRITY_MISMATCH_MESSAGE)


def pack_for_submission(eval_cfg, env, scenario_parameters):

    team_name = eval_cfg["team_name"]
    timestamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

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

    # Last, and deliberately so: the check is advisory, and the socket timeout
    # bounds each blocking operation rather than the whole request, so a slow
    # peer can still drag it out. Running it after the file is on disk means it
    # can no longer cost anyone the run they just simulated.
    _check_evaluate_integrity()


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
