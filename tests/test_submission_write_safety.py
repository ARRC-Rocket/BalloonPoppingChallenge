"""Write-path safety for ``pack_for_submission`` (see issue #59).

Three properties, none of which the format change in #58 should be allowed to
lose:

* the output path stays inside the results directory whatever ``team_name`` says;
* a write that fails part way through leaves no truncated submission and no
  leftover temp file behind;
* two runs in the same second do not land on the same path.

Only ``rocketpy`` is guarded: a missing simulation stack is a legitimate skip,
but a broken import inside this package is a failure and must stay loud.
"""

import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from importlib.util import find_spec
from unittest import mock

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.evaluation.results import utils


class _FakeEnv:
    """The attributes ``pack_for_submission`` reads, and nothing else.

    ``_rocket_flight`` is left as ``None`` on purpose: ``RocketPyEncoder``
    serializes that to ``null``, which keeps the payload small without needing a
    real flight.
    """

    _popped_count = 7
    trajectories = [{"time": 0.0}]
    _balloon_release_at_step = [0]
    _rocket_flight = None
    _balloon_flights = [[[0.0]]]


def _eval_cfg(team_name):
    return {
        "team_name": team_name,
        "team_secret": "secret",
        "agent_module_path": __file__,
        "agent_name": "agent",
        "scenario_number": 0,
    }


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestFilenameSlug(unittest.TestCase):
    """A team name cannot introduce a path component."""

    def test_separators_and_parent_references_are_removed(self):
        for raw_name, expected in [
            ("../../etc/passwd", "etc_passwd"),
            ("..", "team"),
            ("/absolute/path", "absolute_path"),
            (r"C:\Windows\system32", "C_Windows_system32"),
            ("", "team"),
            ("...", "team"),
            ("ok.name-1_2", "ok.name-1_2"),
            ("trailing dot.", "trailing dot"),
        ]:
            with self.subTest(raw_name=raw_name):
                self.assertEqual(utils._filename_slug(raw_name), expected)

    def test_a_readable_name_survives_unchanged(self):
        # A whitelist of ASCII would mangle these, and the filename is how a
        # competitor tells their own submissions apart.
        for raw_name in ["Team Rocket", "ARRC 火箭隊", "チーム", "Équipe"]:
            with self.subTest(raw_name=raw_name):
                self.assertEqual(utils._filename_slug(raw_name), raw_name)

    def test_the_slug_is_never_a_path(self):
        for raw_name in ["../x", "a/b", "a\\b", "/x", "..", "\x00x", "a\nb"]:
            with self.subTest(raw_name=raw_name):
                slug = utils._filename_slug(raw_name)
                self.assertEqual(os.path.basename(slug), slug)
                self.assertFalse(os.path.isabs(slug))
                self.assertNotIn("..", slug)

    def test_a_long_name_is_capped_in_bytes(self):
        # Filesystem name limits are in bytes. Capping characters would let 64
        # four-byte characters past a 255-byte limit once the timestamp and the
        # suffix are added.
        for raw_name in ["x" * 500, "火" * 200, "🎈" * 200]:
            with self.subTest(raw_name=raw_name[:2]):
                slug = utils._filename_slug(raw_name)
                self.assertLessEqual(len(slug.encode("utf-8")), 96)
                # Cutting bytes can split a character; the tail must be dropped
                # rather than left as a partial sequence or a replacement char.
                self.assertNotIn("�", slug)
                self.assertEqual(slug, slug.encode("utf-8").decode("utf-8"))


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestAtomicWrite(unittest.TestCase):
    """The final path only ever holds a complete file."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.out_path = os.path.join(self.directory, "submission.pkl")

    def _leftovers(self):
        return [name for name in os.listdir(self.directory) if name != "submission.pkl"]

    def test_a_successful_write_lands_at_the_final_path(self):
        utils._write_atomically(self.out_path, lambda file: file.write(b"payload"))

        with open(self.out_path, "rb") as file:
            self.assertEqual(file.read(), b"payload")
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_a_failed_write_produces_no_submission(self):
        def write_half_then_fail(file):
            file.write(b"half a subm")
            raise OSError("disk full")

        with self.assertRaises(OSError):
            utils._write_atomically(self.out_path, write_half_then_fail)

        # Writing straight to the final path would leave a truncated file here
        # under a name that looks finished.
        self.assertFalse(os.path.exists(self.out_path))
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_a_failed_write_does_not_destroy_the_previous_submission(self):
        with open(self.out_path, "wb") as file:
            file.write(b"an earlier good submission")

        def fail(_file):
            raise OSError("boom")

        with self.assertRaises(OSError):
            utils._write_atomically(self.out_path, fail)

        with open(self.out_path, "rb") as file:
            self.assertEqual(file.read(), b"an earlier good submission")
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_an_interrupt_also_cleans_up(self):
        def interrupt(_file):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            utils._write_atomically(self.out_path, interrupt)

        self.assertFalse(os.path.exists(self.out_path))
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_a_text_writer_works_too(self):
        """The mode is a parameter because the writer is not always binary.

        ``json.dump`` writes ``str``, and through a binary handle that raises
        ``TypeError: a bytes-like object is required`` after the temp file
        already exists. Measured against the fixed ``"wb"`` this replaces. The
        JSON submission format reuses this helper rather than reimplementing the
        atomicity, so both modes have to hold.
        """
        utils._write_atomically(
            self.out_path,
            lambda file: json.dump({"team": "隊伍"}, file),
            mode="w",
            encoding="utf-8",
        )

        with open(self.out_path, encoding="utf-8") as file:
            self.assertEqual(json.load(file), {"team": "隊伍"})
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_an_fsync_failure_leaves_the_previous_submission_alone(self):
        """Delayed writeback errors surface here, not from write().

        A full disk can be reported at fsync rather than at the writes, which is
        the case this helper exists for, so it needs its own injection rather
        than standing behind the callback-raises test.
        """
        with open(self.out_path, "wb") as file:
            file.write(b"an earlier good submission")

        with mock.patch.object(os, "fsync", side_effect=OSError("no space left")):
            with self.assertRaises(OSError):
                utils._write_atomically(self.out_path, lambda file: file.write(b"new"))

        with open(self.out_path, "rb") as file:
            self.assertEqual(file.read(), b"an earlier good submission")
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")

    def test_a_replace_failure_leaves_the_previous_submission_alone(self):
        # The rename is the other operation this helper introduces. It can fail
        # on permissions, or on Windows with a scanner holding the destination.
        with open(self.out_path, "wb") as file:
            file.write(b"an earlier good submission")

        with mock.patch.object(os, "replace", side_effect=OSError("cannot replace")):
            with self.assertRaises(OSError):
                utils._write_atomically(self.out_path, lambda file: file.write(b"new"))

        with open(self.out_path, "rb") as file:
            self.assertEqual(file.read(), b"an earlier good submission")
        self.assertEqual(self._leftovers(), [], "a temp file was left behind")


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestPackedOutputPath(unittest.TestCase):
    """The real ``pack_for_submission`` composes a path that cannot escape.

    ``_write_atomically`` is replaced by a recorder so this exercises the real
    path composition without writing a submission into the installed package.
    The assertion uses ``os.path.relpath`` against the results directory rather
    than rebuilding the expected filename, so it cannot pass just by repeating
    the implementation.
    """

    def _packed_path(self, team_name):
        recorded = []

        with (
            mock.patch.object(
                utils.urllib.request, "urlopen", side_effect=OSError("offline")
            ),
            mock.patch.object(
                utils,
                "_write_atomically",
                lambda path, _writer, **_kw: recorded.append(path),
            ),
        ):
            utils.pack_for_submission(
                _eval_cfg(team_name), _FakeEnv(), {"scenario": {}}
            )

        self.assertEqual(len(recorded), 1)
        return recorded[0]

    def test_a_hostile_team_name_stays_in_the_results_directory(self):
        results_dir = os.path.dirname(os.path.abspath(utils.__file__))

        for team_name in ["../../../etc", "a/b", "..", r"..\..\x"]:
            with self.subTest(team_name=team_name):
                relative = os.path.relpath(self._packed_path(team_name), results_dir)
                self.assertEqual(os.path.dirname(relative), "")
                self.assertNotIn("..", relative)

    def test_two_runs_in_the_same_millisecond_get_different_names(self):
        """The same instant, not merely the same second.

        An earlier version of this used 1000 and 2000 microseconds, which are two
        different milliseconds, so it only proved that a second-resolution name
        would have collided. Milliseconds narrow that window without closing it:
        ``packed_at`` is read before the integrity check and before serializing a
        few hundred megabytes, so two packers can share a millisecond and finish
        far apart. ``os.replace`` does not refuse an existing destination, so the
        slower one would have replaced a finished submission.
        """
        moment = datetime(2026, 7, 27, 12, 0, 0, 123456, tzinfo=timezone.utc)
        names = []
        for _ in range(2):
            with mock.patch.object(utils, "datetime") as clock:
                clock.now.return_value = moment
                names.append(os.path.basename(self._packed_path("team")))

        self.assertEqual(
            names[0][: len("20260727T120000.123Z")],
            names[1][: len("20260727T120000.123Z")],
            "the timestamps should be identical, or this proves nothing",
        )
        self.assertNotEqual(names[0], names[1])
        self.assertRegex(
            names[0], r"^\d{8}T\d{6}\.\d{3}Z_team_[0-9a-f]{32}_submission\.pkl$"
        )

    def test_two_names_that_slug_the_same_still_get_different_paths(self):
        """The collision the timestamp cannot help with.

        ``a/b`` and ``a\\b`` both normalise to ``a_b``, and any name made only of
        unsafe characters falls back to ``team``, so two different competitors
        can reach the same slug at the same instant.
        """
        moment = datetime(2026, 7, 27, 12, 0, 0, 123456, tzinfo=timezone.utc)
        for first, second in ((r"a/b", "a\\b"), ("   ", "///")):
            with self.subTest(names=(first, second)):
                names = []
                for team_name in (first, second):
                    with mock.patch.object(utils, "datetime") as clock:
                        clock.now.return_value = moment
                        names.append(os.path.basename(self._packed_path(team_name)))

                self.assertEqual(
                    utils._filename_slug(first),
                    utils._filename_slug(second),
                    "these names should slug the same, or this proves nothing",
                )
                self.assertNotEqual(names[0], names[1])

    def test_the_payload_keeps_the_configured_team_name(self):
        # The slug is a filename concern only. The leaderboard reads the name from
        # the payload, so rewriting it there would be a real behaviour change.
        captured = {}

        with (
            mock.patch.object(
                utils.urllib.request, "urlopen", side_effect=OSError("offline")
            ),
            mock.patch.object(
                utils,
                "_write_atomically",
                lambda _path, writer, **_kw: writer(io.BytesIO()),
            ),
            mock.patch.object(
                utils.pickle, "dump", lambda payload, _file: captured.update(payload)
            ),
        ):
            utils.pack_for_submission(
                _eval_cfg("Team Rocket/1"), _FakeEnv(), {"scenario": {}}
            )

        self.assertEqual(captured["team"]["name"], "Team Rocket/1")
        self.assertEqual(captured["leaderboard_info"]["team_name"], "Team Rocket/1")


if __name__ == "__main__":
    unittest.main()
