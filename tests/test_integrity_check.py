"""The evaluate.py integrity check must never cost a competitor a submission.

The check runs at the end of ``pack_for_submission``, after the whole scenario
has been simulated. It only prints a warning, so a DNS failure, a proxy, an
offline machine or a GitHub outage must not be able to abort packing and throw
away a run that took minutes to produce.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import glob
import json
import http.client
import os
import unittest
import urllib.error
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import mock

_STACK_AVAILABLE = find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    import BalloonPoppingGymEnv.evaluation.results.utils as utils


def _fake_env():
    """The smallest object ``pack_for_submission`` accepts."""
    return SimpleNamespace(
        _popped_count=3,
        trajectories=[{"rocket_states": [1.0], "balloon_states": [[0.0] * 6]}],
        _balloon_release_at_step=[0],
        _rocket_flight=None,
        _balloon_flights=[[[0.0]]],
    )


class _IntegrityHelpers:
    """Fixtures only. Deliberately not a TestCase: a TestCase base carrying
    test methods is collected in its own right and its methods then run again in
    every subclass, and an undecorated base also sidesteps the skip that keeps
    this file quiet without the simulation stack."""

    def setUp(self):
        self.results_dir = os.path.dirname(utils.__file__)
        self.agent_file = os.path.join(self.results_dir, "_unittest_agent.py")
        with open(self.agent_file, "w", encoding="utf-8") as handle:
            handle.write("# unit-test agent source\n")
        self.eval_cfg = {
            "team_name": "unittest_team",
            "team_secret": "s3cr3t",
            "agent_name": "A",
            "scenario_number": 0,
            "agent_module_path": self.agent_file,
        }
        self.created = []

    def tearDown(self):
        for path in self.created:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.agent_file):
            os.remove(self.agent_file)

    def _pack(self, urlopen_side_effect):
        before = set(glob.glob(os.path.join(self.results_dir, "*_submission.*")))
        with mock.patch.object(
            utils.urllib.request, "urlopen", side_effect=urlopen_side_effect
        ):
            utils.pack_for_submission(self.eval_cfg, _fake_env(), {"scenario": {}})
        new = set(glob.glob(os.path.join(self.results_dir, "*_submission.*"))) - before
        self.assertEqual(len(new), 1, "packing should still produce a submission")
        path = new.pop()
        self.created.append(path)
        return path

    class _Response:
        """Minimal stand-in for the urlopen context manager."""

        def __init__(
            self,
            payload,
            declared=None,
            status=200,
            url=None,
            chunked=False,
            headers=None,
        ):
            self._payload = payload
            # Defaults are the shape of a healthy response, so every other case
            # has to opt in rather than be reached by omission.
            self.status = status
            self.chunked = chunked
            self._url = utils.INTEGRITY_CHECK_URL if url is None else url
            if headers is not None:
                self.headers = headers
            else:
                self.headers = {
                    "Content-Length": str(
                        len(payload) if declared is None else declared
                    )
                }

        def geturl(self):
            return self._url

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, amount=None):
            return self._payload if amount is None else self._payload[:amount]

    def _local_bytes(self):
        local = os.path.join(os.path.dirname(self.results_dir), "evaluate.py")
        with open(local, "rb") as handle:
            return handle.read()

    def _pack_and_capture(self, response=None, **response_kwargs):
        """Pack once with a stubbed response and return everything printed."""
        if response is None:
            response = self._Response(b"", **response_kwargs)
        before = set(glob.glob(os.path.join(self.results_dir, "*_submission.*")))
        with mock.patch.object(utils.urllib.request, "urlopen", return_value=response):
            with mock.patch("builtins.print") as printed:
                utils.pack_for_submission(self.eval_cfg, _fake_env(), {"scenario": {}})
        new = set(glob.glob(os.path.join(self.results_dir, "*_submission.*"))) - before
        # Every path here is advisory, so the submission must exist regardless.
        self.assertEqual(len(new), 1, "packing should still produce a submission")
        self.created.append(new.pop())
        return " ".join(str(call) for call in printed.call_args_list)

    def _pack_path_after(self, call):
        before = set(glob.glob(os.path.join(self.results_dir, "*_submission.*")))
        call()
        new = set(glob.glob(os.path.join(self.results_dir, "*_submission.*"))) - before
        self.assertEqual(len(new), 1)
        return new.pop()


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestIntegrityCheckFailsOpen(_IntegrityHelpers, unittest.TestCase):
    """Every expected failure still leaves a submission behind."""

    def test_network_failure_still_produces_a_submission(self):
        """Before the fix the URLError propagated and the run was lost."""
        path = self._pack(urllib.error.URLError("name resolution failed"))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["team"]["name"], "unittest_team")

    def test_timeout_still_produces_a_submission(self):
        self._pack(TimeoutError("timed out"))

    def test_http_error_still_produces_a_submission(self):
        self._pack(
            urllib.error.HTTPError(
                url="https://example.invalid",
                code=503,
                msg="unavailable",
                hdrs=None,
                fp=None,
            )
        )

    def test_the_request_carries_a_timeout(self):
        """An unbounded request could hang the run instead of failing fast."""
        recorded = {}

        def fake_urlopen(url, timeout=None):
            recorded["timeout"] = timeout
            raise urllib.error.URLError("stop here")

        with mock.patch.object(utils.urllib.request, "urlopen", fake_urlopen):
            self.created.append(
                self._pack_path_after(
                    lambda: utils.pack_for_submission(
                        self.eval_cfg, _fake_env(), {"scenario": {}}
                    )
                )
            )
        self.assertIsNotNone(recorded.get("timeout"), "urlopen needs a timeout")
        self.assertEqual(recorded["timeout"], utils.INTEGRITY_CHECK_TIMEOUT)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestIntegrityCheckBounds(_IntegrityHelpers, unittest.TestCase):
    """The check is bounded, and every expected failure leaves packing alone."""

    def _pack_with_response(self, payload):
        return self._pack_returning(lambda *a, **k: self._Response(payload))

    def _pack_returning(self, fake_urlopen):
        before = set(glob.glob(os.path.join(self.results_dir, "*_submission.*")))
        with mock.patch.object(utils.urllib.request, "urlopen", fake_urlopen):
            utils.pack_for_submission(self.eval_cfg, _fake_env(), {"scenario": {}})
        new = set(glob.glob(os.path.join(self.results_dir, "*_submission.*"))) - before
        self.assertEqual(len(new), 1, "packing should still produce a submission")
        path = new.pop()
        self.created.append(path)
        return path

    def test_reading_the_local_file_can_fail_without_losing_the_submission(self):
        """The local read used to sit outside the guarded block."""
        real_open = open

        def failing_open(path, *args, **kwargs):
            if str(path).endswith("evaluate.py"):
                raise PermissionError("evaluate.py is not readable")
            return real_open(path, *args, **kwargs)

        before = set(glob.glob(os.path.join(self.results_dir, "*_submission.*")))
        with mock.patch("builtins.open", failing_open):
            utils.pack_for_submission(self.eval_cfg, _fake_env(), {"scenario": {}})
        new = set(glob.glob(os.path.join(self.results_dir, "*_submission.*"))) - before
        self.assertEqual(len(new), 1)
        self.created.append(new.pop())

    def test_an_oversized_reference_reports_that_it_cannot_check(self):
        """A capped read leaves a truncated body, which must not be reported as
        tampering: the honest answer is that the check could not run."""
        oversized = b"x" * (utils.INTEGRITY_CHECK_MAX_BYTES + 10)
        with mock.patch("builtins.print") as printed:
            self._pack_with_response(oversized)
        said = " ".join(str(c) for c in printed.call_args_list)
        self.assertIn("unexpectedly large", said)
        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)

    def test_the_remote_body_read_has_a_size_cap(self):
        recorded = {}

        class Recording(self._Response):
            def read(inner, amount=None):
                recorded["amount"] = amount
                return b""

        self._pack_returning(lambda *a, **k: Recording(b""))
        self.assertEqual(recorded["amount"], utils.INTEGRITY_CHECK_MAX_BYTES + 1)

    def test_a_matching_reference_is_quiet(self):
        with mock.patch("builtins.print") as printed:
            self._pack_with_response(self._local_bytes())
        said = " ".join(str(c) for c in printed.call_args_list)
        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)

    def test_a_mismatching_reference_warns(self):
        with mock.patch("builtins.print") as printed:
            self._pack_with_response(self._local_bytes() + b"\n# tampered\n")
        said = " ".join(str(c) for c in printed.call_args_list)
        self.assertIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)

    def test_a_blocked_hash_still_produces_a_submission(self):
        """A FIPS-restricted build refuses the digest with ValueError."""
        with mock.patch.object(
            utils.hashlib, "sha256", side_effect=ValueError("disabled by FIPS policy")
        ):
            self._pack(urllib.error.URLError("unused"))

    def test_an_oversized_local_file_still_produces_a_submission(self):
        real_open = open

        class _Huge:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, amount=None):
                size = utils.INTEGRITY_CHECK_MAX_BYTES + 10
                return b"x" * (size if amount is None else min(amount, size))

        def huge_open(path, *args, **kwargs):
            if str(path).endswith("evaluate.py"):
                return _Huge()
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", huge_open), mock.patch("builtins.print") as p:
            self._pack_returning(lambda *a, **k: self._Response(b""))
        said = " ".join(str(c) for c in p.call_args_list)
        self.assertIn("local copy is unexpectedly large", said)

    def test_a_protocol_error_still_produces_a_submission(self):
        """BadStatusLine is HTTPException but not OSError, unlike the others."""
        self._pack(http.client.BadStatusLine("garbage"))

    def test_an_incomplete_response_is_not_reported_as_tampering(self):
        """A peer can announce a length and close early without raising.

        Hashing the short body would accuse the competitor of editing a file
        they never touched.
        """
        with mock.patch("builtins.print") as printed:
            self._pack_returning(
                lambda *a, **k: self._Response(b"short", declared=10_000)
            )
        said = " ".join(str(c) for c in printed.call_args_list)
        self.assertIn("arrived incomplete", said)
        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestOnlyAFull200Counts(_IntegrityHelpers, unittest.TestCase):
    """urlopen accepts the whole 2xx range, and most of it is not the file.

    ``HTTPErrorProcessor`` raises only outside ``200 <= code < 300``, so a 203
    rewritten by an intermediary, a 204 with no body, and a 206 carrying one
    slice all arrive as ordinary successful responses. Hashing any of them
    reports a mismatch for a file the competitor never touched, which is the one
    thing this check must not do.
    """

    def _pack_with(self, payload, **response_kwargs):
        return self._pack_and_capture(self._Response(payload, **response_kwargs))

    def test_a_partial_response_is_not_a_mismatch(self):
        # The dangerous one: a 206 declares the length of its own partial body
        # and matches it, so the completeness check passes and the truncated
        # bytes reach the digest comparison.
        official = self._local_bytes()
        partial = official[: len(official) // 2]

        said = self._pack_with(partial, status=206)

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)
        self.assertIn("unexpected HTTP status 206", said)

    def test_a_transformed_response_is_not_a_mismatch(self):
        said = self._pack_with(b"# rewritten by a proxy\n", status=203)

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)
        self.assertIn("unexpected HTTP status 203", said)

    def test_an_empty_response_is_not_a_mismatch(self):
        said = self._pack_with(b"", status=204)

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)

    def test_a_missing_status_attribute_is_not_a_mismatch(self):
        # Whatever the object is, if it cannot say it is a 200 the check is
        # unavailable rather than a finding about the local file.
        response = self._Response(b"anything")
        del response.status

        said = self._pack_and_capture(response)

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)

    # Two methods rather than two packs in one. The submission filename resolves
    # to the second, so packing twice inside one test reuses the name and the
    # "one new file" assertion sees none. Separate methods get a fresh tearDown.
    def test_a_matching_200_is_quiet(self):
        # The control for the four above: refusing every response outright would
        # pass them all.
        said = self._pack_with(self._local_bytes(), status=200)

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)
        self.assertNotIn("unexpected HTTP status", said)

    def test_a_differing_200_still_reports_a_mismatch(self):
        said = self._pack_with(b"# actually different\n", status=200)

        self.assertIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheBodyMustBeDelimited(_IntegrityHelpers, unittest.TestCase):
    """Completeness comes from the framing, and only two kinds carry it.

    A chunked body ends at a terminator http.client validates. A declared
    length can be compared against what arrived. A response with neither is
    delimited by the connection closing, and RFC 9112 says such a body cannot
    be told apart from one cut short: read() returns whatever turned up and
    nothing raises.
    """

    def test_a_close_delimited_partial_body_is_not_a_mismatch(self):
        official = self._local_bytes()
        response = self._Response(official[: len(official) // 2], headers={})

        said = self._pack_and_capture(response)

        self.assertIn("without a length", said)
        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)

    def test_a_chunked_response_needs_no_declared_length(self):
        # The control. Refusing everything without a Content-Length would pass
        # the case above while breaking a perfectly good chunked transfer.
        response = self._Response(
            self._local_bytes(), chunked=True, headers={"Transfer-Encoding": "chunked"}
        )

        said = self._pack_and_capture(response)

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)
        self.assertNotIn("without a length", said)

    def test_a_chunked_response_that_differs_still_reports_a_mismatch(self):
        response = self._Response(
            b"# actually different\n",
            chunked=True,
            headers={"Transfer-Encoding": "chunked"},
        )

        self.assertIn(
            utils.INTEGRITY_MISMATCH_MESSAGE, self._pack_and_capture(response)
        )


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheResponseMustComeFromTheRightPlace(_IntegrityHelpers, unittest.TestCase):
    """urlopen follows redirects, so a 200 is not proof of origin."""

    def test_a_redirect_to_somewhere_else_is_not_a_mismatch(self):
        response = self._Response(
            b"<html>sign in to your network</html>",
            url="https://captive.example.invalid/login",
        )

        said = self._pack_and_capture(response)

        self.assertIn("redirected to", said)
        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)

    def test_a_coded_body_is_not_a_mismatch(self):
        # The client asks for identity and cannot decode anything else, so a
        # coded body would be hashed in its coded form.
        response = self._Response(
            b"\x1f\x8b\x08 not really gzip",
            headers={"Content-Length": "20", "Content-Encoding": "gzip"},
        )

        said = self._pack_and_capture(response)

        self.assertIn("Content-Encoding", said)
        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestUnusableFramingIsNotAMismatch(_IntegrityHelpers, unittest.TestCase):
    """A Content-Length that will not parse says the framing cannot be trusted."""

    def _pack_with_declared(self, payload, declared):
        return self._pack_and_capture(self._Response(payload, declared=declared))

    def test_a_non_numeric_content_length(self):
        said = self._pack_with_declared(b"# different\n", "not-a-number")

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)
        self.assertIn("unusable Content-Length", said)

    def test_a_negative_content_length(self):
        # Parses, so the old code took it as a length and found len(raw) >= -5.
        said = self._pack_with_declared(b"# different\n", -5)

        self.assertNotIn(utils.INTEGRITY_MISMATCH_MESSAGE, said)
        self.assertIn("unusable Content-Length", said)


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestSubmissionIsDurableFirst(_IntegrityHelpers, unittest.TestCase):
    """The advisory check runs only once the result is safely on disk.

    Every other test here inspects the file after pack_for_submission returns,
    which cannot tell the two orderings apart: moving the check back in front of
    the write leaves them all green, because each failure mode fails open and
    the file is still written afterwards.
    """

    def test_the_submission_is_complete_when_the_check_runs(self):
        observations = []
        pattern = os.path.join(self.results_dir, "*_submission.*")
        # Only files this call creates. Submissions are gitignored, so a working
        # tree that has produced one before already has files matching the
        # pattern, and globbing the directory outright would count those, or read
        # one of them instead of the file under test. CI checks out clean, which
        # is why an unbounded glob stays green there.
        before = set(glob.glob(pattern))

        def inspect_at_call_time():
            # Appended, not overwritten. Keying a dict meant a wrong
            # implementation that checked once before the write and once after
            # it passed: the second observation replaced the first, and the
            # assertions only ever saw the successful one.
            created = set(glob.glob(pattern)) - before
            observation = {"count": len(created)}
            if len(created) == 1:
                with open(next(iter(created)), encoding="utf-8") as handle:
                    observation["payload"] = json.load(handle)
            observations.append(observation)

        with mock.patch.object(
            utils, "_check_evaluate_integrity", side_effect=inspect_at_call_time
        ):
            utils.pack_for_submission(self.eval_cfg, _fake_env(), {"scenario": {}})
            self.created.extend(set(glob.glob(pattern)) - before)

        self.assertEqual(
            len(observations), 1, "the check must run once, after the write"
        )
        self.assertEqual(observations[0]["count"], 1)
        self.assertEqual(observations[0]["payload"]["team"]["name"], "unittest_team")

        # Written, closed and complete, not just created.
        self.assertIn("balloon_world_data", observations[0]["payload"])


if __name__ == "__main__":
    unittest.main()
