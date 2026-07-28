"""The job that gates a merge has to test what it says it tests.

CI used to run ``uv lock --check`` and then ``pip install -r
requirements-dev.txt``. That file carries no pins, so the check passed and said
nothing about what was then installed. Measured on the develop it was fixed
from: the lockfile pinned matplotlib 3.10.9, numpy 2.4.5 and scipy 1.17.1, and
pip resolved 3.11.1, 2.5.1 and 1.18.0.

Moving the gate onto the lockfile introduced a second way to be wrong, which is
why these are asserted rather than left to the comments in the workflow. uv
picks its interpreter from ``.python-version``, which pins 3.14, and ignores
whatever ``setup-python`` put on PATH. Without ``UV_PYTHON`` the matrix leg
named 3.10 builds a 3.14 environment and passes, so the floor stops being
tested while the run stays green.

Reads the workflow as data. No simulation stack needed.
"""

import unittest
from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
)

# The job whose result gates a merge, and the one that is allowed to install
# whatever pip resolves because it cannot block anything.
GATING_JOB = "test"
EARLY_WARNING_JOB = "latest-dependencies"

# pyproject claims >=3.10, and competitors run this on their own machines.
SUPPORTED_PYTHON = {"3.10", "3.14"}


def _commands(job):
    """Every shell line the job runs, without comments.

    Line by line and comment-stripped, so a mention inside a ``#`` line or an
    ``echo`` cannot satisfy an assertion about what the job does.
    """
    lines = []
    for step in job.get("steps", []):
        for line in step.get("run", "").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def _runs(job, fragment):
    return any(fragment in line for line in _commands(job))


class TestTheGatingJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))["jobs"]

    def test_the_gating_job_exists(self):
        """Or every assertion below holds over a job that is not there."""
        self.assertIn(GATING_JOB, self.jobs, f"jobs are {sorted(self.jobs)}")

    def test_it_installs_from_the_lockfile(self):
        self.assertTrue(_runs(self.jobs[GATING_JOB], "uv sync --locked"))

    def test_it_does_not_install_unpinned_requirements(self):
        self.assertFalse(_runs(self.jobs[GATING_JOB], "requirements-dev.txt"))
        self.assertFalse(_runs(self.jobs[GATING_JOB], "pip install -r"))

    def test_uv_is_told_which_interpreter_the_leg_is_for(self):
        """``.python-version`` pins 3.14 and uv follows it over PATH.

        So without this the leg named 3.10 builds a 3.14 environment, passes,
        and the floor #75 exists to test is no longer tested by anything.
        """
        self.assertEqual(
            self.jobs[GATING_JOB].get("env", {}).get("UV_PYTHON"),
            "${{ matrix.python-version }}",
        )

    def test_it_checks_the_interpreter_it_actually_got(self):
        """The environment variable is the fix; this is the proof it worked.

        Both halves. Looking only for ``sys.version_info`` passed a step that
        read the version and then did nothing with it.
        """
        commands = _commands(self.jobs[GATING_JOB])

        self.assertTrue(any("sys.version_info" in line for line in commands))
        self.assertTrue(
            any("assert" in line and "want" in line for line in commands),
            "the interpreter is reported but never checked",
        )

    def test_it_runs_the_tests_in_the_environment_it_built(self):
        """``uv run`` re-resolves before running unless told not to, which would
        put the thing under test back out of the lockfile's hands."""
        self.assertTrue(_runs(self.jobs[GATING_JOB], "uv run --no-sync pytest"))

    def test_it_covers_every_supported_python(self):
        versions = self.jobs[GATING_JOB]["strategy"]["matrix"]["python-version"]

        self.assertEqual(set(map(str, versions)), SUPPORTED_PYTHON)

    def test_it_can_still_block(self):
        """A gate that never fails is not a gate.

        ``assertNotEqual(value, True)`` was not enough:
        ``continue-on-error: ${{ true }}`` parses as a non-empty string, so it
        passed that while GitHub evaluated it to true.
        """
        self.assertIn(
            self.jobs[GATING_JOB].get("continue-on-error"),
            (None, False),
            "the gating job cannot fail a run",
        )


class TestTheEarlyWarningJob(unittest.TestCase):
    """Installing the newest dependencies is still worth doing, elsewhere.

    Matplotlib 3.11 took out the default renderer for anyone installing without
    the lockfile, and CI only saw it because CI installed that way. Moving the
    gate to the lockfile would have removed the only thing that caught it.
    """

    @classmethod
    def setUpClass(cls):
        cls.jobs = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))["jobs"]

    def test_the_unpinned_install_is_still_exercised(self):
        self.assertIn(EARLY_WARNING_JOB, self.jobs, f"jobs are {sorted(self.jobs)}")
        self.assertTrue(_runs(self.jobs[EARLY_WARNING_JOB], "requirements-dev.txt"))

    def test_it_actually_runs_the_tests(self):
        """Installing and not running anything would satisfy the rest of this."""
        self.assertTrue(_runs(self.jobs[EARLY_WARNING_JOB], "pytest tests/"))

    def test_it_covers_every_supported_python(self):
        """pip does not resolve the same stack on each.

        The lockfile itself carries numpy 2.2.6 below Python 3.11 and 2.4.5
        above it, so covering only the newest leaves the floor unwatched.
        """
        versions = self.jobs[EARLY_WARNING_JOB]["strategy"]["matrix"]["python-version"]

        self.assertEqual(set(map(str, versions)), SUPPORTED_PYTHON)

    def test_it_does_not_block(self):
        """It installs whatever PyPI has, so a release elsewhere would otherwise
        stop unrelated work from merging.

        Required to be the literal boolean. ``continue-on-error: ${{ false }}``
        is a non-empty string to a YAML parser, so a truthiness check passes it
        while GitHub evaluates the expression to false and the job blocks again.
        """
        self.assertIs(self.jobs[EARLY_WARNING_JOB].get("continue-on-error"), True)


if __name__ == "__main__":
    unittest.main()
