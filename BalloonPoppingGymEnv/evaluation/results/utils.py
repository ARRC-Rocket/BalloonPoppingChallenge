import hashlib
import http.client
import json
import logging
import os
import pickle
import re
import tempfile
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4

from rocketpy._encoders import RocketPyEncoder

logger = logging.getLogger(__name__)

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
    "still saved. A checkout from a revision other than current main can report "
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

            # urlopen follows redirects on its own, so a proxy or captive portal
            # that lands on its own 200 page arrives here looking successful.
            final_url = response.geturl()
            if final_url != INTEGRITY_CHECK_URL:
                print(f"Could not check evaluate.py: redirected to {final_url}")
                return

            # http.client asks for identity and cannot decode anything else, so
            # a coded body would be hashed in its coded form.
            encoding = response.headers.get("Content-Encoding")
            if encoding not in (None, "", "identity"):
                print(f"Could not check evaluate.py: Content-Encoding {encoding}")
                return

            chunked = bool(getattr(response, "chunked", False))
            declared = response.headers.get("Content-Length")
            # One byte over the cap, so an oversized body is detectable.
            raw = response.read(INTEGRITY_CHECK_MAX_BYTES + 1)

        if len(raw) > INTEGRITY_CHECK_MAX_BYTES:
            print(
                "Could not check evaluate.py: the reference copy is unexpectedly large"
            )
            return

        # Completeness has to come from the framing, and there are only two
        # kinds that carry it. A chunked body ends at a terminator http.client
        # validates, raising on a short stream. A declared length can be
        # compared against what arrived. A response with neither is delimited
        # by the connection closing, and RFC 9112 is explicit that such a body
        # cannot be told apart from one cut short mid-transfer: read() hands
        # back whatever turned up and nothing raises. Hashing that would accuse
        # a competitor of editing a file they never touched.
        if chunked:
            pass
        elif declared is None:
            print(
                "Could not check evaluate.py: the reference copy arrived without a "
                "length to check it against"
            )
            return
        else:
            # A length that will not parse, or parses negative, says the framing
            # cannot be trusted, which is the same conclusion.
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


def _submission_filename(packed_at, team_name, run_id):
    """Name a submission so two runs cannot land on the same path.

    The timestamp sorts and the slug identifies; neither makes the name unique.
    Milliseconds narrow the window rather than closing it, and ``packed_at`` is
    taken before the integrity check and before serializing a few hundred
    megabytes, so two packers can share a millisecond and finish minutes apart.
    ``os.replace`` does not refuse an existing destination, so the second one
    would atomically replace the first competitor's finished submission, which is
    a worse failure than the truncated file the atomic write is here to prevent.

    Slugs collide too, which the timestamp cannot help with: ``a/b`` and ``a\\b``
    both normalise to ``a_b``, and a name of only unsafe characters falls back to
    ``team``. ``run_id`` is what actually makes the path unique.
    """
    stamp = f"{packed_at:%Y%m%dT%H%M%S}.{packed_at.microsecond // 1000:03d}Z"
    return f"{stamp}_{_filename_slug(team_name)}_{run_id}_submission.pkl"


def _write_atomically(out_path, write_payload, mode="wb", encoding=None):
    """Write via a temp file in the same directory, then ``os.replace`` it.

    A scenario-1 submission runs to a few hundred megabytes. Writing straight to
    the final path means a disk-full or a killed process leaves a truncated file
    under a name that looks finished, with nothing to distinguish it from a good
    submission. Same-directory ``os.replace`` is atomic on POSIX and Windows, so
    the final path only ever holds a complete file.

    ``write_payload`` receives the open handle. ``mode`` and ``encoding`` are
    arguments rather than fixed at ``"wb"`` because the writer is not always
    binary: ``json.dump`` writes ``str``, and a text writer through a binary
    handle raises ``TypeError: a bytes-like object is required``. Keeping both
    here means the JSON submission format can reuse this rather than reimplement
    the atomicity.

    The flush and fsync are what make the rename meaningful for the case this is
    here for: a disk that fills up surfaces as an error from one of the two,
    rather than as a renamed file holding nothing. Full power-loss durability
    would additionally need an fsync on the parent directory, which is not
    claimed and is not what #59 was about.

    One deliberate difference from a plain ``open``: ``mkstemp`` creates the file
    0600 rather than at the umask default, and the mode survives the rename. The
    payload carries the team secret, so keeping it owner-only is the better end
    state on a shared machine.
    """
    handle = None
    temp_path = None
    try:
        handle, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(out_path) or ".", prefix=".partial_", suffix=".tmp"
        )
        # `closefd=False` so this function is the descriptor's only owner. The
        # alternative, handing ownership to the file object, cannot be written
        # safely: `os.fdopen` closes the descriptor itself on some failures
        # (an unknown encoding is one, measured), and then the handle a cleanup
        # path still holds names whatever has since been given that number.
        with os.fdopen(handle, mode, encoding=encoding, closefd=False) as file:
            write_payload(file)
            file.flush()
            os.fsync(file.fileno())
        # Before the rename rather than after: Windows refuses to rename a file
        # that still has an open handle, and closing here rather than in the
        # cleanup below keeps the success path from depending on it.
        os.close(handle)
        handle = None
        os.replace(temp_path, out_path)
    except BaseException:
        # Including KeyboardInterrupt: a stray .partial_ left behind would be a
        # worse outcome than the truncated file this exists to prevent. A
        # SIGKILL still can, which no handler can cover; what it cannot leave is
        # a file under the finished name.
        if handle is not None:
            try:
                os.close(handle)
            except OSError:
                pass
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                # Said rather than swallowed. The original error is what raises,
                # but a partial file nobody knows about is worse than one that
                # was named in the log.
                try:
                    logger.warning(
                        "Could not remove the incomplete submission at %s: %s",
                        temp_path,
                        cleanup_error,
                    )
                except Exception:  # noqa: BLE001 - a diagnostic cannot be the failure
                    # Handlers and filters are the application's code and can
                    # raise. Letting one out here would replace the disk full
                    # the caller needs to see with a logging error.
                    pass
        raise


def build_submission_payload(eval_cfg, env, scenario_parameters, packed_at):
    """What goes into a submission, without deciding how it is written.

    Separate from `pack_for_submission` so the contents can be tested without a
    serializer: capturing the payload by mocking `pickle.dump` would tie what a
    submission contains to how it is encoded, and #58 replaces the encoder.
    """
    team_name = eval_cfg["team_name"]
    timestamp = f"{packed_at:%Y%m%dT%H%M%SZ}"

    # Read agent source
    agent_module_path = os.fspath(eval_cfg["agent_module_path"])
    with open(agent_module_path, "r", encoding="utf-8") as f:
        agent_module_file = f.read()

    return {
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
            # "balloon_flights": env._balloon_flights,  # deliberately omitted, see issue #57.
        },
        "agent_info": {
            "eval_cfg": eval_cfg,
            "agent_module_file": agent_module_file,
        },
    }


def pack_for_submission(eval_cfg, env, scenario_parameters):

    team_name = eval_cfg["team_name"]
    packed_at = datetime.now(timezone.utc)
    submission = build_submission_payload(eval_cfg, env, scenario_parameters, packed_at)

    # Save submission. The filename carries milliseconds and a path-safe team
    # name, so two runs in the same second no longer overwrite each other. The
    # payload above keeps the second-resolution timestamp and the configured team
    # name untouched, since those are the copies the leaderboard reads.
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        _submission_filename(packed_at, team_name, uuid4().hex),
    )
    _write_atomically(out_path, lambda file: pickle.dump(submission, file))

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
