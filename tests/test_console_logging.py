"""The output an entry point actually produces.

The existing logging test checks that two loggers exist and that ``close``
emits something. That leaves the part a user sees untested: whether the CLI
still prints the same line, whether anything is printed at all when
``evaluate_scenario`` is called from a script or a notebook, and whether the
engine's own records end up mixed into it.

Nothing here simulates anything. An earlier version of this file drove the real
CLI against the shipped config, which renders with Matplotlib, writes a
trajectory file into the package directory, makes a network request to GitHub
and writes a submission carrying the team secret, all to check a formatter. It
went red in CI on a Matplotlib version difference that has nothing to do with
logging.
"""

import ast
import importlib.util
import io
import json
import logging
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from BalloonPoppingGymEnv.console_logging import (
    CONSOLE_HANDLER_NAME,
    PACKAGE_LOGGER_NAME,
    configure_console_logging,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SCRIPT = REPO_ROOT / "doc" / "examples" / "run_env_agent.py"
NOTEBOOK = REPO_ROOT / "doc" / "examples" / "evaluate_scenario_colab.ipynb"


class _LoggingStateMixin:
    """Put the package logger back exactly as it was, handlers included."""

    def preserve_logging_state(self):
        """Detach what was there, rather than snapshot and overwrite.

        Putting a saved list back is not a restoration if something closed one
        of its handlers in the meantime, and that is exactly what
        ``configure_console_logging`` does to a console handler it is replacing.
        A process that had already configured console logging would get its
        handler back closed. Detaching first means the helper only ever sees
        handlers this test installed.
        """
        package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        handlers = list(package_logger.handlers)
        level = package_logger.level
        propagate = package_logger.propagate
        for handler in handlers:
            package_logger.removeHandler(handler)

        def restore():
            for existing in list(package_logger.handlers):
                package_logger.removeHandler(existing)
            for handler in handlers:
                package_logger.addHandler(handler)
            package_logger.setLevel(level)
            package_logger.propagate = propagate

        self.addCleanup(restore)
        return package_logger


class TestTheConsoleFormat(_LoggingStateMixin, unittest.TestCase):
    def setUp(self):
        self.package_logger = self.preserve_logging_state()
        self.stream = io.StringIO()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

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

    def test_a_child_left_at_debug_does_not_leak_through(self):
        """Why the threshold is on the handler as well as the logger.

        A record is filtered by the level of the logger it was emitted on;
        propagation then hands it to every ancestor handler without rechecking
        any ancestor logger's level. With the handler left at NOTSET, a
        descendant at DEBUG puts debug records on a competitor's stdout.
        Measured before the fix: the debug line was printed.
        """
        configure_console_logging(level=logging.INFO, stream=self.stream)
        child = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.envs.balloon_world")
        previous = child.level
        self.addCleanup(child.setLevel, previous)
        child.setLevel(logging.DEBUG)

        child.debug("internal state nobody asked for")
        child.info("Total reward: 7")

        self.assertEqual(self.stream.getvalue(), "Total reward: 7\n")

    def test_configuring_twice_does_not_double_the_output(self):
        # A notebook cell gets rerun.
        configure_console_logging(stream=self.stream)
        configure_console_logging(stream=self.stream)

        logging.getLogger(PACKAGE_LOGGER_NAME).info("once")

        self.assertEqual(self.stream.getvalue(), "once\n")

    def test_a_handler_the_host_installed_is_left_alone(self):
        """Only this module's own handler is replaced.

        An application embedding the environment can have its own file, JSON or
        audit handler on this logger. Clearing the list outright discards it, and
        it never comes back.
        """
        host_stream = io.StringIO()
        host_handler = logging.StreamHandler(host_stream)
        host_handler.set_name("host.audit")
        self.package_logger.addHandler(host_handler)

        configure_console_logging(stream=self.stream)
        configure_console_logging(stream=self.stream)

        self.assertIn(host_handler, self.package_logger.handlers)
        owned = [
            handler
            for handler in self.package_logger.handlers
            if handler.get_name() == CONSOLE_HANDLER_NAME
        ]
        self.assertEqual(len(owned), 1, "the console handler should be replaced, once")

        logging.getLogger(PACKAGE_LOGGER_NAME).info("Total reward: 7")
        self.assertEqual(self.stream.getvalue(), "Total reward: 7\n")
        self.assertEqual(host_stream.getvalue(), "Total reward: 7\n")

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

    def test_importing_the_package_configures_no_logging(self):
        """The library default, in a process that has done nothing else.

        The previous version of this cleared the package and root handlers and
        then checked that nothing printed, which removed the very configuration
        it was meant to detect: adding a configure call at import time left it
        passing. A fresh interpreter is the only place this contract is
        observable.
        """
        report = Path(self.directory) / "state.json"
        code = (
            "import json, logging, sys\n"
            "import BalloonPoppingGymEnv.evaluation.evaluate  # noqa: F401\n"
            "logger = logging.getLogger('BalloonPoppingGymEnv')\n"
            "logger.info('a record no entry point asked to see')\n"
            f"open({str(report)!r}, 'w').write(json.dumps("
            "{'handlers': [type(h).__name__ for h in logger.handlers],"
            " 'root_handlers': [type(h).__name__ for h in logging.getLogger().handlers]}))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertEqual(completed.stdout, "", "importing the package wrote to stdout")
        self.assertEqual(completed.stderr, "", "importing the package wrote to stderr")
        state = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(state["handlers"], [], "import installed a handler")
        self.assertEqual(state["root_handlers"], [])


class TestTheCommandLine(_LoggingStateMixin, unittest.TestCase):
    """The CLI's own output, with the simulation replaced.

    Driving the real config would render, write a trajectory file, request the
    reference copy of evaluate.py over the network and write a submission, none
    of which this is about, and any of which can turn it red for its own
    reasons.
    """

    def setUp(self):
        self.preserve_logging_state()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config_path = Path(directory.name) / "eval_cfg.yaml"
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "scenario_number": 0,
                    "render_mode": None,
                    "agent_module_path": "./BalloonPoppingGymEnv/agents/example_agents.py",
                    "agent_class_name": "AttitudeRateControlAgent",
                    "agent_name": "Attitude Rate Control",
                    "agent_kwargs": {"launch_time": 1},
                    "team_name": "unittest_team",
                    "team_secret": "not-a-real-secret",
                    "leaderboard_submission": False,
                }
            ),
            encoding="utf-8",
        )

    def _run_cli(self):
        from BalloonPoppingGymEnv.evaluation import evaluate

        def fake_evaluate_scenario(*_args, **_kwargs):
            # Stands in for a run, emitting the two records a real one does.
            evaluate.logger.info(
                "Scenario %s evaluation completed with agent '%s'.", 0, "A"
            )
            evaluate.logger.info("Total reward: %s", 7)
            return object(), object(), {}

        captured = io.StringIO()
        with mock.patch.object(evaluate, "_load_agent_class", return_value=object):
            with mock.patch.object(
                evaluate, "evaluate_scenario", fake_evaluate_scenario
            ):
                with mock.patch(
                    "BalloonPoppingGymEnv.console_logging.sys.stdout", captured
                ):
                    with redirect_stdout(captured):
                        evaluate.main([str(self.config_path)])
        return captured.getvalue()

    def test_the_command_line_output_is_exactly_the_two_lines(self):
        # The whole of stdout, not one line matched by prefix. The previous
        # version allowed any amount of extra output and only rejected the
        # substring "INFO:", so an integrity warning or a traceback preamble
        # would have passed.
        self.assertEqual(
            self._run_cli(),
            "Scenario 0 evaluation completed with agent 'A'.\nTotal reward: 7\n",
        )

    def test_a_missing_config_path_still_explains_itself(self):
        from BalloonPoppingGymEnv.evaluation import evaluate

        with self.assertRaises(ValueError) as raised:
            evaluate.main([])

        self.assertIn("Configuration file path is required", str(raised.exception))


class TestTheEntryPointsConfigureIt(unittest.TestCase):
    """Every caller of evaluate_scenario, not only the CLI.

    The notebook is still read with ``ast``, because a notebook cell cannot be
    executed here without running the scenario it configures. The example is
    executed, which is strictly stronger.

    evaluate_scenario logs its completion and score lines, so a caller that does
    not configure logging shows nothing. That is what happened to the notebook
    and to run_for_evaluation when the prints became logger calls.

    Read with ``ast`` rather than by searching the text. A substring search
    passes on a call inside a comment, in dead code, or placed after the
    evaluation it was supposed to precede.
    """

    @staticmethod
    def _call_names(node):
        return [
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        ]

    def test_the_example_configures_logging_before_evaluating(self):
        """Run it, rather than read it.

        Reading the AST proves a call node exists and is lexically first. It
        cannot tell that from a call under ``if False:``. Executing the function
        with both ends replaced records what actually happens.
        """
        spec = importlib.util.spec_from_file_location(
            "_run_env_agent_under_test", EXAMPLE_SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        order = []
        with mock.patch(
            "BalloonPoppingGymEnv.console_logging.configure_console_logging",
            side_effect=lambda *a, **k: order.append("configure"),
        ):
            with mock.patch(
                "BalloonPoppingGymEnv.evaluation.evaluate.evaluate_scenario",
                side_effect=lambda *a, **k: order.append("evaluate"),
            ):
                module.run_for_evaluation()

        self.assertEqual(order, ["configure", "evaluate"])

    def test_the_notebook_configures_logging_before_evaluating(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        source = next(cell for cell in cells if "evaluate_scenario(" in cell)
        tree = ast.parse(source)
        names = self._call_names(tree)

        self.assertIn("configure_console_logging", names)
        self.assertLess(
            names.index("configure_console_logging"), names.index("evaluate_scenario")
        )

    def test_the_notebook_binds_the_result_rather_than_displaying_it(self):
        """Left as the cell's value, the returned tuple renders the environment."""
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "evaluate_scenario(" in "".join(cell["source"])
        )
        tree = ast.parse(source)

        bare = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "evaluate_scenario"
        ]
        self.assertEqual(bare, [], "the call's result is the cell's displayed value")


if __name__ == "__main__":
    unittest.main()
