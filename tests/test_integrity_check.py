"""The evaluate.py integrity check must never cost a competitor a submission.

The check runs at the end of ``pack_for_submission``, after the whole scenario
has been simulated. It only prints a warning, so a DNS failure, a proxy, an
offline machine or a GitHub outage must not be able to abort packing and throw
away a run that took minutes to produce.

Only ``rocketpy`` is guarded below: a missing simulation stack is a legitimate
skip, but a broken import inside this package is a failure and must stay loud.
"""

import glob
import os
import pickle
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


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestIntegrityCheckFailsOpen(unittest.TestCase):
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

    def test_network_failure_still_produces_a_submission(self):
        """Before the fix the URLError propagated and the run was lost."""
        path = self._pack(urllib.error.URLError("name resolution failed"))
        with open(path, "rb") as handle:
            self.assertEqual(pickle.load(handle)["team"]["name"], "unittest_team")

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

    def _pack_path_after(self, call):
        before = set(glob.glob(os.path.join(self.results_dir, "*_submission.*")))
        call()
        new = set(glob.glob(os.path.join(self.results_dir, "*_submission.*"))) - before
        self.assertEqual(len(new), 1)
        return new.pop()


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestIntegrityCheckBounds(TestIntegrityCheckFailsOpen):
    """The check is bounded, and every expected failure leaves packing alone."""

    class _Response:
        """Minimal stand-in for the urlopen context manager."""

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, amount=None):
            return self._payload if amount is None else self._payload[:amount]

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

    def _local_bytes(self):
        local = os.path.join(os.path.dirname(self.results_dir), "evaluate.py")
        with open(local, "rb") as handle:
            return handle.read()

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
        self.assertNotIn("should not be modified", said)

    def test_the_read_is_bounded(self):
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
        self.assertNotIn("should not be modified", said)

    def test_a_mismatching_reference_warns(self):
        with mock.patch("builtins.print") as printed:
            self._pack_with_response(self._local_bytes() + b"\n# tampered\n")
        said = " ".join(str(c) for c in printed.call_args_list)
        self.assertIn("should not be modified", said)


if __name__ == "__main__":
    unittest.main()
