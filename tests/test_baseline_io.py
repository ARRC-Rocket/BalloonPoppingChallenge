"""Writing a baseline must never leave a bad one under the real filename.

The regenerators are run by hand, rarely, and the failure they can cause is the
hardest kind to notice: a baseline that looks fine and is wrong. Neither the
strict encoding nor the atomic replace had a test.
"""

import json
import math
import os
import stat
import tempfile
import unittest
from unittest import mock

from tests.baselines import baseline_io
from tests.baselines.baseline_io import NEW_BASELINE_MODE, write_baseline


class TestWriteBaseline(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.path = os.path.join(self.directory, "baseline.json")

    def _leftovers(self):
        return [n for n in os.listdir(self.directory) if n != "baseline.json"]

    def test_a_baseline_round_trips(self):
        write_baseline({"popped_count": 10, "positions": [[1.0, 2.0, 3.0]]}, self.path)

        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["popped_count"], 10)
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_the_file_ends_with_a_newline(self):
        write_baseline({"a": 1}, self.path)

        with open(self.path, encoding="utf-8") as handle:
            self.assertTrue(handle.read().endswith("\n"))

    def test_a_non_finite_value_is_refused(self):
        # A diverged trajectory would otherwise be written as a bare NaN token,
        # which is not JSON, and the next run would compare against it.
        with self.assertRaises(ValueError):
            write_baseline({"positions": [[1.0, math.nan, 3.0]]}, self.path)

    def test_a_refused_write_leaves_no_file_at_all(self):
        with self.assertRaises(ValueError):
            write_baseline({"positions": [[math.inf]]}, self.path)

        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_a_refused_write_does_not_destroy_the_previous_baseline(self):
        write_baseline({"popped_count": 10}, self.path)

        with self.assertRaises(ValueError):
            write_baseline({"popped_count": math.nan}, self.path)

        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["popped_count"], 10)
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_a_bare_filename_still_works(self):
        # dirname is "" here, which mkstemp resolves against the working
        # directory. Kept because a caller may pass a bare name, not because the
        # empty string needs handling: I checked, and mkstemp(dir="") works.
        previous = os.getcwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, previous)

        write_baseline({"a": 1}, "baseline.json")

        self.assertTrue(os.path.exists(os.path.join(self.directory, "baseline.json")))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(os.name == "nt", "POSIX permission semantics")
class TestTheReplacementKeepsThePermissions(unittest.TestCase):
    """Replacing a file is not overwriting it, and the mode follows the file.

    ``mkstemp`` creates owner-only and ``os.replace`` moves that file into place,
    so without care every regeneration narrows a ``0664`` baseline to ``0600``.
    The repository's committed baselines are ``0664``, so this is not
    hypothetical. The submission writer elsewhere in the project *wants* owner-only
    because its payload holds the team secret; a baseline does not.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.path = os.path.join(self.directory, "baseline.json")

    def _mode(self):
        return stat.S_IMODE(os.stat(self.path).st_mode)

    def test_replacing_a_baseline_preserves_its_mode(self):
        write_baseline({"old": True}, self.path)
        os.chmod(self.path, 0o664)

        write_baseline({"new": True}, self.path)

        self.assertEqual(self._mode(), 0o664)

    def test_an_unusual_mode_is_preserved_too(self):
        # Not just the one value the repository happens to use.
        write_baseline({"old": True}, self.path)
        os.chmod(self.path, 0o640)

        write_baseline({"new": True}, self.path)

        self.assertEqual(self._mode(), 0o640)

    def test_the_new_baseline_policy_is_group_and_world_readable(self):
        """The policy itself, written out once.

        Every other assertion here compares against NEW_BASELINE_MODE, so
        changing the constant moves both sides and nothing notices. The
        submission writer elsewhere in this project is deliberately owner-only
        because its payload carries the team secret; a committed baseline is an
        ordinary readable artifact and should not quietly acquire that policy.
        """
        self.assertEqual(NEW_BASELINE_MODE, 0o644)

    def test_a_new_baseline_is_readable(self):
        write_baseline({"new": True}, self.path)

        self.assertEqual(self._mode(), NEW_BASELINE_MODE)

    def test_a_restrictive_umask_does_not_narrow_a_new_baseline(self):
        """The policy is stated, not inherited from the process.

        chmod sets permissions outright, so the umask never applies. Consulting
        it would mean setting and restoring a process-wide value, which is the
        race below.
        """
        previous = os.umask(0o077)
        self.addCleanup(os.umask, previous)

        write_baseline({"new": True}, self.path)

        self.assertEqual(self._mode(), NEW_BASELINE_MODE)

    def test_writing_a_baseline_never_touches_the_process_umask(self):
        """The umask is process-wide, not per thread.

        Reading it means setting it and putting it back, and any other thread
        creating a file in that window gets the wrong permissions. Measured with
        the process at 022 and the window widened: an unrelated file opened with
        mode 0666 was created 0600 instead of 0644.
        """
        with mock.patch.object(
            baseline_io.os,
            "umask",
            side_effect=AssertionError("write_baseline must not touch the umask"),
        ):
            write_baseline({"new": True}, self.path)

        self.assertEqual(self._mode(), NEW_BASELINE_MODE)


class TestTheFailurePathsTheHelperDocuments(unittest.TestCase):
    """The two guarantees the cleanup handler exists for, neither of them tested.

    The existing tests all stop inside ``json.dump`` with a ``ValueError``, which
    an ``except Exception`` would catch just as well. Turning ``BaseException``
    into ``Exception`` left all six of them passing while breaking the interrupt
    cleanup the comment promises.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.path = os.path.join(self.directory, "baseline.json")

    def _leftovers(self):
        return [n for n in os.listdir(self.directory) if n != "baseline.json"]

    def test_an_interrupt_after_a_partial_write_cleans_up(self):
        def interrupting_dump(_baseline, temp_file, **_kwargs):
            temp_file.write('{"partial":')
            raise KeyboardInterrupt

        with mock.patch.object(baseline_io.json, "dump", interrupting_dump):
            with self.assertRaises(KeyboardInterrupt):
                write_baseline({"a": 1}, self.path)

        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self._leftovers(), [])

    def test_an_interrupt_does_not_disturb_an_existing_baseline(self):
        write_baseline({"old": True}, self.path)

        def interrupting_dump(_baseline, temp_file, **_kwargs):
            temp_file.write('{"partial":')
            raise KeyboardInterrupt

        with mock.patch.object(baseline_io.json, "dump", interrupting_dump):
            with self.assertRaises(KeyboardInterrupt):
                write_baseline({"new": True}, self.path)

        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"old": True})
        self.assertEqual(self._leftovers(), [])

    def test_a_replace_failure_preserves_the_previous_baseline(self):
        # The other end of the write: the JSON is complete and valid, and the
        # move itself fails. Windows can refuse this when a reader holds the
        # destination open, and the temp file must not be left behind either.
        write_baseline({"old": True}, self.path)

        with mock.patch.object(
            baseline_io.os, "replace", side_effect=PermissionError("busy")
        ):
            with self.assertRaises(PermissionError):
                write_baseline({"new": True}, self.path)

        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"old": True})
        self.assertEqual(self._leftovers(), [])
