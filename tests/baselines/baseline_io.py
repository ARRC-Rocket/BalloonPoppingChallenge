"""Writing a golden-master baseline, shared by both regenerators.

Kept here rather than duplicated in each script: the two regenerators already
diverge in what they collect, and the part that decides whether a half-written
baseline can end up under the real filename should not be one of them.
"""

import json
import os
import stat
import tempfile

# Used only when creating a baseline that does not exist yet. An existing file
# keeps its own mode. Committed baselines are ordinary readable artifacts, unlike
# the submission files written elsewhere in the project, which are deliberately
# owner-only because they carry the team secret.
NEW_BASELINE_MODE = 0o644


def write_baseline(baseline, output_path):
    """Write ``baseline`` as strict JSON, replacing the old file only on success.

    ``json.dump`` writes bare ``NaN`` and ``Infinity`` for non-finite floats.
    RFC 8259 has no such tokens, so a diverged trajectory would land in the
    baseline as something most readers refuse, and the next comparison would run
    against it. ``allow_nan=False`` turns that into an error while the previous
    baseline is still intact.

    The write goes through a temp file in the same directory followed by
    ``os.replace``, so a reader never sees a partially written replacement and a
    regeneration killed part way through cannot leave a truncated file under the
    real name. Writing in place would allow both, with nothing downstream able to
    tell.

    Two consequences of replacing rather than overwriting, both handled here:

    ``mkstemp`` creates its file owner-only, and ``os.replace`` moves that file
    rather than the bytes inside the old one, so the destination would inherit
    ``0600``. The repository's own baselines are ``0664``, so every regeneration
    would have quietly narrowed them. The existing mode is carried over, and a
    baseline created from nothing gets ``NEW_BASELINE_MODE`` under the umask.

    If ``output_path`` is a symlink this replaces the link itself and leaves its
    target alone, where writing in place would have followed it. The repository's
    baselines are ordinary files, so this is a note rather than a problem.
    """
    # dirname is "" for a bare filename, which mkstemp resolves against the
    # working directory, so it needs no special case.
    handle, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path), prefix=".partial_", suffix=".json"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            json.dump(baseline, temp_file, indent=2, allow_nan=False)
            temp_file.write("\n")
        os.chmod(temp_path, _mode_to_apply(output_path))
        os.replace(temp_path, output_path)
    except BaseException:
        # Including KeyboardInterrupt: a stray .partial_ would be a worse outcome
        # than the truncated file this exists to prevent.
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _mode_to_apply(output_path):
    """The mode the replacement should carry: the old file's, or the default."""
    try:
        return stat.S_IMODE(os.stat(output_path).st_mode)
    except FileNotFoundError:
        return NEW_BASELINE_MODE & ~_current_umask()


def _current_umask():
    """Read the umask without leaving it changed.

    ``os.umask`` is the only way to read it, and it always sets. Restore it
    immediately; a new baseline is rare enough that the brief window does not
    matter here, and the alternative is ignoring the umask entirely.
    """
    mask = os.umask(0o077)
    os.umask(mask)
    return mask
