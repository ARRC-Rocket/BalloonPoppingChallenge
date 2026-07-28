"""The output an entry point actually produces.

The existing logging test checks that two loggers exist and that ``close``
emits something. That leaves the part a user sees untested: whether the CLI
still prints the same line, whether anything is printed at all when
``evaluate_scenario`` is called from a script or a notebook, and whether the
engine's own INFO records end up mixed into it.
"""

import io
import json
import logging
import subprocess
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

from BalloonPoppingGymEnv.console_logging import (
    PACKAGE_LOGGER_NAME,
    configure_console_logging,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_STACK_AVAILABLE = find_spec("rocketpy") is not None


class TestTheConsoleFormat(unittest.TestCase):
    def setUp(self):
        package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        previous = (
            list(package_logger.handlers),
            package_logger.level,
            package_logger.propagate,
        )

        def restore():
            package_logger.handlers[:] = previous[0]
            package_logger.setLevel(previous[1])
            package_logger.propagate = previous[2]

        self.addCleanup(restore)
        self.stream = io.StringIO()

    def test_a_record_arrives_as_its_bare_message(self):
        # basicConfig's default is levelname:name:message, which turned
        # "Total reward: 7" into "INFO:__main__:Total reward: 7".
        configure_console_logging(stream=self.stream)

        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.evaluation.evaluate").info(
            "Total reward: %s", 7
        )

        self.assertEqual(self.stream.getvalue(), "Total reward: 7\n")

    def test_another_package_does_not_reach_the_stream(self):
        """Scoped to this package, not the root logger.

        basicConfig on the root opens the same stream to every dependency that
        propagates a record. RocketPy is adding module loggers of its own, so a
        competitor's score would arrive inside the engine's chatter.
        """
        configure_console_logging(stream=self.stream)

        logging.getLogger("rocketpy.simulation.flight").info("solver chatter")

        self.assertEqual(self.stream.getvalue(), "")

    def test_configuring_twice_does_not_double_the_output(self):
        # A notebook cell gets rerun.
        configure_console_logging(stream=self.stream)
        configure_console_logging(stream=self.stream)

        logging.getLogger(PACKAGE_LOGGER_NAME).info("once")

        self.assertEqual(self.stream.getvalue(), "once\n")

    def test_a_host_root_handler_does_not_double_the_line(self):
        """Why propagate is turned off.

        Something else owning the process can already have a root handler: a
        notebook kernel, a harness, an application embedding the environment.
        Left propagating, every package record reaches both that handler and
        this one, and the competitor sees their score twice.
        """
        host_stream = io.StringIO()
        root = logging.getLogger()
        previous_handlers, previous_level = list(root.handlers), root.level
        self.addCleanup(root.setLevel, previous_level)
        self.addCleanup(root.handlers.__setitem__, slice(None), previous_handlers)
        root.handlers[:] = [logging.StreamHandler(host_stream)]
        root.setLevel(logging.INFO)

        configure_console_logging(stream=self.stream)
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.envs").info("Total reward: 7")

        self.assertEqual(self.stream.getvalue(), "Total reward: 7\n")
        self.assertEqual(host_stream.getvalue(), "")

    def test_nothing_is_printed_before_an_entry_point_asks(self):
        """The default for a library, and the reason the examples must call this.

        A logger with no handler and no configuration emits nothing at INFO, so
        evaluate_scenario called from a script or a notebook was silent.
        """
        package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        package_logger.handlers[:] = []
        package_logger.setLevel(logging.NOTSET)

        with self.assertNoLogs(PACKAGE_LOGGER_NAME, level="WARNING"):
            package_logger.info("not shown at the default threshold")


class TestTheEntryPointsConfigureIt(unittest.TestCase):
    """Every caller of evaluate_scenario, not only the CLI.

    evaluate_scenario logs its completion and score lines, so a caller that does
    not configure logging shows nothing. That is what happened to the notebook
    and to run_for_evaluation when the prints became logger calls.
    """

    def test_the_example_script_configures_logging(self):
        source = (REPO_ROOT / "doc" / "examples" / "run_env_agent.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("configure_console_logging()", source)

    def test_the_notebook_configures_logging_and_binds_the_result(self):
        notebook = json.loads(
            (
                REPO_ROOT / "doc" / "examples" / "evaluate_scenario_colab.ipynb"
            ).read_text(encoding="utf-8")
        )
        cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        calling = next(cell for cell in cells if "evaluate_scenario(" in cell)

        self.assertIn("configure_console_logging()", calling)
        # Left as the cell's value, the returned tuple renders the whole
        # environment where the score line should be.
        self.assertNotRegex(calling, r"(?m)^evaluate_scenario\(")


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheCommandLineOutput(unittest.TestCase):
    """The exact stdout, through a real subprocess.

    Nothing else here can see the format the CLI actually produces: the
    configuration lives under ``if __name__ == "__main__"``, so importing the
    module does not run it.
    """

    def test_the_score_line_is_unprefixed(self):
        completed = subprocess.run(
            [
                sys.executable,
                "BalloonPoppingGymEnv/evaluation/evaluate.py",
                "BalloonPoppingGymEnv/evaluation/configs/example_eval_cfg.yaml",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        score_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("Total reward:")
        ]
        self.assertEqual(len(score_lines), 1, completed.stdout[-2000:])
        # Not "INFO:__main__:Total reward: N", and not "INFO:...:" anywhere.
        self.assertNotIn("INFO:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
