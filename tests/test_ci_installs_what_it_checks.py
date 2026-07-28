"""The job that gates a merge has to install the versions the lockfile pins.

CI used to run ``uv lock --check`` and then ``pip install -r
requirements-dev.txt``. That file carries no pins, so the check passed and said
nothing about what was then installed. Measured on the develop it was fixed
from: the lockfile pinned matplotlib 3.10.9, numpy 2.4.5 and scipy 1.17.1, and
pip resolved 3.11.1, 2.5.1 and 1.18.0.

Two things follow. A green run was not a statement about any particular set of
versions, so a local run and a CI run were not testing the same software. And a
release on PyPI could turn the branch red with nothing changed here.

This is a one-line thing to undo by accident, and nothing else would notice, so
it is asserted rather than left to the comment in the workflow.

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


def _run_steps(job):
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


class TestTheGatingJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        cls.jobs = cls.workflow["jobs"]

    def test_the_gating_job_exists(self):
        """Or every assertion below holds over a job that is not there."""
        self.assertIn(GATING_JOB, self.jobs, f"jobs are {sorted(self.jobs)}")

    def test_it_installs_from_the_lockfile(self):
        self.assertIn("uv sync --locked", _run_steps(self.jobs[GATING_JOB]))

    def test_it_does_not_install_unpinned_requirements(self):
        """The exact thing that was there before.

        ``requirements-dev.txt`` is two editable installs and no versions, so
        installing from it resolves whatever PyPI has that day.
        """
        commands = _run_steps(self.jobs[GATING_JOB])

        self.assertNotIn("requirements-dev.txt", commands)
        self.assertNotIn("pip install -r", commands)

    def test_it_runs_the_tests_inside_that_environment(self):
        """Installing into the uv environment and then running the system
        interpreter's pytest would test neither set of versions."""
        commands = _run_steps(self.jobs[GATING_JOB])

        self.assertIn("uv run pytest", commands)

    def test_it_can_still_block(self):
        """The opposite failure: a gate that never fails is not a gate."""
        self.assertNotEqual(
            self.jobs[GATING_JOB].get("continue-on-error"),
            True,
            "the gating job cannot fail a run",
        )


class TestTheEarlyWarningJob(unittest.TestCase):
    """Installing the newest dependencies is still worth doing, elsewhere.

    Matplotlib 3.11 took out the default renderer for anyone on the README's pip
    path, and CI only saw it because CI was on that path. Moving the gate to the
    lockfile would have removed the only thing that caught it, so the pip path
    keeps running under its own name.
    """

    @classmethod
    def setUpClass(cls):
        cls.jobs = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))["jobs"]

    def test_the_pip_path_is_still_exercised(self):
        self.assertIn(EARLY_WARNING_JOB, self.jobs, f"jobs are {sorted(self.jobs)}")
        self.assertIn("requirements-dev.txt", _run_steps(self.jobs[EARLY_WARNING_JOB]))

    def test_it_does_not_block(self):
        """It installs whatever PyPI has, so a release elsewhere would otherwise
        stop unrelated work from merging."""
        self.assertTrue(self.jobs[EARLY_WARNING_JOB].get("continue-on-error"))


if __name__ == "__main__":
    unittest.main()
