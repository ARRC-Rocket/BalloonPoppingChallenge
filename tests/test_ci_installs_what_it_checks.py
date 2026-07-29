"""The gating job has to test what it says it tests. ``uv lock --check`` beside
an unpinned ``pip install`` passed while the lockfile pinned matplotlib 3.10.9,
numpy 2.4.5, scipy 1.17.1 and pip installed 3.11.1, 2.5.1, 1.18.0."""

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
    """Every shell line the job runs, without comments or echoes. With the
    install and the pytest line replaced by ``echo`` of the same strings, all
    twelve tests passed over a job that installed nothing and ran nothing."""
    lines = []
    for step in job.get("steps", []):
        for line in step.get("run", "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.split()[0] in ("echo", "printf", ":", "true"):
                continue
            lines.append(stripped)
    return lines


def _runs(job, fragment):
    return any(fragment in line for line in _commands(job))


class TestWhatTheWorkflowMustNotGiveAway(unittest.TestCase):
    """Properties that a later edit would undo without anything noticing."""

    def setUp(self):
        self.jobs = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))["jobs"]

    def test_no_job_holds_a_secret_in_its_environment(self):
        """A secret in a job's `env` is handed to uv, pip, pytest and anything
        they start. Only the step that needs it should get it, through `with`."""
        for name, job in self.jobs.items():
            for key, value in (job.get("env") or {}).items():
                with self.subTest(job=name, variable=key):
                    text = str(value)
                    # `secrets.X != ''` carries whether a secret exists, which is
                    # what a step's `if` can read. The value itself is the leak.
                    self.assertFalse(
                        "secrets." in text and "!=" not in text,
                        f"{name}.env.{key} puts a secret in the job environment",
                    )

    def test_every_job_has_a_timeout(self):
        """The default ceiling is six hours, and a non-finite command has already
        left this project's solver refusing to return."""
        for name, job in self.jobs.items():
            with self.subTest(job=name):
                self.assertIsInstance(job.get("timeout-minutes"), int)

    def test_a_run_with_no_coverage_cannot_upload_nothing(self):
        """`warn` lets a run that produced no coverage report look fine."""
        uploads = [
            step
            for step in self.jobs[GATING_JOB]["steps"]
            if "upload-artifact" in str(step.get("uses", ""))
        ]
        self.assertTrue(uploads)
        for step in uploads:
            with self.subTest(step=step.get("name")):
                self.assertEqual(step["with"].get("if-no-files-found"), "error")


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
        """``.python-version`` pins 3.14 and uv follows it over PATH, so without
        this the leg named 3.10 builds a 3.14 environment, passes, and the floor
        #75 exists to test is no longer tested by anything."""
        self.assertEqual(
            self.jobs[GATING_JOB].get("env", {}).get("UV_PYTHON"),
            "${{ matrix.python-version }}",
        )

    def test_it_checks_the_interpreter_it_actually_got(self):
        """Both halves. Looking only for ``sys.version_info`` passed a step that
        read the version and then did nothing with it."""
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
        """A gate that never fails is not a gate. ``assertNotEqual(value, True)``
        is not enough: ``continue-on-error: ${{ true }}`` parses as a non-empty
        string, so it passes that while GitHub evaluates it to true."""
        self.assertIn(
            self.jobs[GATING_JOB].get("continue-on-error"),
            (None, False),
            "the gating job cannot fail a run",
        )

    def test_no_step_of_it_swallows_its_own_failure(self):
        """The same escape one level down, where the job-level check cannot see
        it. ``continue-on-error: true`` on the pytest step alone left every test
        here passing, and a red suite would have reported the job green."""
        for step in self.jobs[GATING_JOB].get("steps", []):
            with self.subTest(step=step.get("name", step.get("uses", "?"))):
                self.assertIn(step.get("continue-on-error"), (None, False))

    def test_nothing_can_switch_it_off(self):
        """``if: false`` on the job, and the subtler ``if: github.event_name ==
        'schedule'``, each left every test here passing while the job stopped
        running. actionlint reports only the literal ``false``."""
        self.assertIsNone(
            self.jobs[GATING_JOB].get("if"),
            "the gating job runs unconditionally or it is not a gate",
        )

    def test_it_runs_the_tests_that_only_run_when_asked(self):
        """The golden masters are behind ``BPC_RUN_SLOW_TESTS``, and deleting
        that one line is silent: every test here still passes and the run goes
        from 325 passed with 2 skipped to 309 with 18, the scenarios among them."""
        environment = self.jobs[GATING_JOB].get("env", {})

        self.assertEqual(str(environment.get("BPC_RUN_SLOW_TESTS", "")), "1")


class TestTheEarlyWarningJob(unittest.TestCase):
    """Installing the newest dependencies is still worth doing, elsewhere:
    matplotlib 3.11 took out the default renderer for anyone installing without
    the lockfile, and CI only saw it because CI installed that way."""

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
        """pip does not resolve the same stack on each. The lockfile itself
        carries numpy 2.2.6 below Python 3.11 and 2.4.5 above it, so covering
        only the newest leaves the floor unwatched."""
        versions = self.jobs[EARLY_WARNING_JOB]["strategy"]["matrix"]["python-version"]

        self.assertEqual(set(map(str, versions)), SUPPORTED_PYTHON)

    def test_it_does_not_block(self):
        """It installs whatever PyPI has, so a release elsewhere would otherwise
        stop unrelated work from merging. The literal boolean, because
        ``${{ false }}`` is a non-empty string to a YAML parser."""
        self.assertIs(self.jobs[EARLY_WARNING_JOB].get("continue-on-error"), True)


if __name__ == "__main__":
    unittest.main()


class TestWhatCountsAsSomethingTheJobDoes(unittest.TestCase):
    """``_commands`` is the lens every assertion above looks through, so it is
    tested rather than trusted. A lens that quietly widens turns all of them
    into statements about nothing."""

    def test_a_real_command_is_kept(self):
        job = {"steps": [{"run": "uv sync --locked --extra dev"}]}

        self.assertEqual(_commands(job), ["uv sync --locked --extra dev"])

    def test_a_commented_command_is_not_one(self):
        job = {"steps": [{"run": "# uv sync --locked --extra dev"}]}

        self.assertEqual(_commands(job), [])

    def test_an_echoed_command_is_not_one(self):
        """With the install and the pytest line replaced by ``echo`` of the same
        strings, every test in this file passed."""
        for line in (
            'echo "uv sync --locked --extra dev"',
            "printf 'uv run --no-sync pytest'",
            ": uv sync --locked",
        ):
            with self.subTest(line=line):
                self.assertEqual(_commands({"steps": [{"run": line}]}), [])
