"""Writing a golden-master baseline, shared by both regenerators.

Kept here rather than duplicated in each script: the two regenerators already
diverge in what they collect, and the part that decides whether a half-written
baseline can end up under the real filename should not be one of them.
"""

import json
import os
import stat
import tempfile

# Used only when creating a baseline that does not exist yet, and applied with
# chmod rather than filtered through the umask. An existing file keeps its own
# mode. Committed baselines are ordinary readable artifacts, unlike
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
    handle = None
    temp_path = None
    try:
        handle, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(output_path), prefix=".partial_", suffix=".json"
        )
        # `closefd=False` so this function is the descriptor's only owner, and
        # `mkstemp` inside the try so it has one at all. Same shape and the same
        # reasons as `_write_atomically`: `os.fdopen` closes the descriptor
        # itself on some failures, so a cleanup path cannot also own it.
        with os.fdopen(handle, "w", encoding="utf-8", closefd=False) as temp_file:
            json.dump(baseline, temp_file, indent=2, allow_nan=False)
            temp_file.write("\n")
        os.close(handle)
        handle = None
        os.chmod(temp_path, _mode_to_apply(output_path))
        os.replace(temp_path, output_path)
    except BaseException:
        # Including KeyboardInterrupt: a stray .partial_ would be a worse outcome
        # than the truncated file this exists to prevent.
        if handle is not None:
            try:
                os.close(handle)
            except OSError:
                pass
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


def _mode_to_apply(output_path):
    """The mode the replacement should carry: the old file's, or the default.

    The umask is deliberately not consulted. Reading it means setting it and
    putting it back, and it is process-wide rather than per thread, so any
    other thread creating a file in that window gets the wrong permissions.
    Measured: with the process at 022 and one thread reading the umask this
    way, an unrelated file opened with mode 0666 was created 0600 instead of
    0644.

    ``chmod`` sets permissions outright, so there is nothing to emulate. These
    are committed repository artifacts and the policy for a new one is stated
    above; an existing one keeps whatever it had.
    """
    try:
        return stat.S_IMODE(os.stat(output_path).st_mode)
    except FileNotFoundError:
        return NEW_BASELINE_MODE
