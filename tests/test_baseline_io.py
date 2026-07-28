"""Writing a baseline must never leave a bad one under the real filename.

The regenerators are run by hand, rarely, and the failure they can cause is the
hardest kind to notice: a baseline that looks fine and is wrong. Neither the
strict encoding nor the atomic replace had a test.
"""

import json
import math
import os
import tempfile
import unittest

from tests.baselines.baseline_io import write_baseline


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
