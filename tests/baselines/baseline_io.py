"""Writing a golden-master baseline, shared by both regenerators.

Kept here rather than duplicated in each script: the two regenerators already
diverge in what they collect, and the part that decides whether a half-written
baseline can end up under the real filename should not be one of them.
"""

import json
import os
import tempfile


def write_baseline(baseline, output_path):
    """Write ``baseline`` as strict JSON, replacing the old file only on success.

    ``json.dump`` writes bare ``NaN`` and ``Infinity`` for non-finite floats.
    RFC 8259 has no such tokens, so a diverged trajectory would land in the
    baseline as something most readers refuse, and the next comparison would run
    against it. ``allow_nan=False`` turns that into an error while the previous
    baseline is still intact.

    The write goes through a temp file in the same directory followed by
    ``os.replace``, which is atomic on POSIX and Windows. Writing in place would
    let a regeneration killed part way through leave a truncated file under the
    real name, and nothing downstream would be able to tell.
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
        os.replace(temp_path, output_path)
    except BaseException:
        # Including KeyboardInterrupt: a stray .partial_ would be a worse outcome
        # than the truncated file this exists to prevent.
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
