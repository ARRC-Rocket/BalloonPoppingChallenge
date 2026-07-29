"""The output an entry point actually produces.

The existing logging test checks that two loggers exist and that ``close``
emits something. That leaves the part a user sees untested: whether the CLI
still prints the same line, whether anything is printed at all when
``evaluate_scenario`` is called from a script or a notebook, and whether the
engine's own records end up mixed into it.

Most of this simulates nothing. The two termination records at the bottom are
the exception: reaching them needs a real episode, and one of them needs a real
flight. An earlier version of this file drove the real
CLI against the shipped config, which renders with Matplotlib, writes a
trajectory file into the package directory, makes a network request to GitHub
and writes a submission carrying the team secret, all to check a formatter. It
went red in CI on a Matplotlib version difference that has nothing to do with
logging.
"""

import ast
import importlib.util
from importlib.util import find_spec
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
# find_spec answers "is it installed", which is the only case that justifies a
# skip; the imports inside the tests then fail loudly on a broken stack.
_STACK_AVAILABLE = find_spec("rocketpy") is not None
EXAMPLE_SCRIPT = REPO_ROOT / "doc" / "examples" / "run_env_agent.py"
NOTEBOOK = REPO_ROOT / "doc" / "examples" / "evaluate_scenario_colab.ipynb"


class _ClosureTrackingHandler(logging.StreamHandler):
    """Records whether it was closed, which the stream cannot be asked.

    ``Handler.close`` marks the handler closed and drops its name registration
    but leaves the stream alone, so writing through a closed StreamHandler still
    works and cannot be used to tell the two apart.
    """

    def __init__(self, stream):
        super().__init__(stream)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


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

    def test_a_hosts_lower_level_is_not_raised(self):
        """Preserving a handler is not the same as preserving the policy.

        An application that set DEBUG on this logger did so to feed its own
        handler. Setting the logger to INFO would silence that handler while
        leaving it attached, which looks like preservation and is not. The
        console threshold lives on the console handler, so the logger can stay
        wherever the host put it.
        """
        host_stream = io.StringIO()
        host_handler = logging.StreamHandler(host_stream)
        host_handler.set_name("host.debug_file")
        host_handler.setLevel(logging.DEBUG)
        self.package_logger.addHandler(host_handler)
        self.package_logger.setLevel(logging.DEBUG)

        configure_console_logging(level=logging.INFO, stream=self.stream)

        self.assertEqual(self.package_logger.level, logging.DEBUG)
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.envs").debug("host wants this")
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.envs").info("Total reward: 7")

        self.assertEqual(host_stream.getvalue(), "host wants this\nTotal reward: 7\n")
        # The console still shows only INFO and above.
        self.assertEqual(self.stream.getvalue(), "Total reward: 7\n")

    def test_a_level_name_is_accepted_like_logging_accepts_one(self):
        """logging takes "INFO"; the arithmetic below it did not.

        With the package logger already carrying an explicit level, comparing a
        string against an int raised TypeError, and by then the new handler was
        attached and the old one closed. The caller got an exception and a
        half-configured logger.
        """
        self.package_logger.setLevel(logging.DEBUG)

        configure_console_logging(level="INFO", stream=self.stream)

        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.envs").debug("not on the console")
        logging.getLogger(f"{PACKAGE_LOGGER_NAME}.envs").info("Total reward: 7")
        self.assertEqual(self.stream.getvalue(), "Total reward: 7\n")

    def test_an_inherited_level_is_not_raised_either(self):
        """NOTSET on a non-root logger means ask the ancestors, not no level.

        Reading the raw attribute, finding NOTSET and setting INFO raised the
        threshold on a package logger whose effective level was DEBUG through
        the root. Measured before the fix: effective went DEBUG to INFO.
        """
        self.package_logger.setLevel(logging.NOTSET)
        root = logging.getLogger()
        previous = root.level
        self.addCleanup(root.setLevel, previous)
        root.setLevel(logging.DEBUG)
        self.assertEqual(self.package_logger.getEffectiveLevel(), logging.DEBUG)

        configure_console_logging(level=logging.INFO, stream=self.stream)

        self.assertEqual(self.package_logger.getEffectiveLevel(), logging.DEBUG)

    def test_a_logger_with_no_level_gets_the_requested_one(self):
        # The root level is part of this assertion, not ambient. The level
        # written is min(effective, requested), and effective is inherited from
        # the root once this logger is at NOTSET, so a process that had left the
        # root at DEBUG would legitimately produce DEBUG here. Measured: this
        # test failed exactly that way with the root at DEBUG.
        root = logging.getLogger()
        self.addCleanup(root.setLevel, root.level)
        root.setLevel(logging.WARNING)
        self.package_logger.setLevel(logging.NOTSET)

        configure_console_logging(level=logging.INFO, stream=self.stream)

        self.assertEqual(self.package_logger.level, logging.INFO)

    def test_a_level_logging_rejects_leaves_the_logger_untouched(self):
        """The half-configured state the normalisation was moved for.

        Asserted on the logger rather than on the exception, because the
        exception was never the problem: the old order attached the new handler
        and closed the old one first, so a caller that caught the error was left
        with console output going somewhere it did not ask for. Measured with
        the check moved back below the swap: every other test here stayed green.
        """
        installed = _ClosureTrackingHandler(io.StringIO())
        installed.set_name(CONSOLE_HANDLER_NAME)
        self.package_logger.addHandler(installed)
        self.package_logger.setLevel(logging.WARNING)
        self.package_logger.propagate = True

        with self.assertRaises(ValueError):
            configure_console_logging(level="NOT_A_LEVEL", stream=self.stream)

        self.assertEqual(self.package_logger.handlers, [installed])
        self.assertEqual(self.package_logger.level, logging.WARNING)
        self.assertTrue(self.package_logger.propagate)
        # Asked of the handler, not of its stream. An earlier version of this
        # emitted a record and checked it arrived, which proves nothing:
        # StreamHandler.close inherits Handler.close, which does not touch the
        # stream, so emit keeps working on a closed handler. Measured.
        self.assertFalse(installed.was_closed, "the owned handler was closed")
        self.assertRegistryStillPointsAt(installed)

    def assertRegistryStillPointsAt(self, handler):
        """The part of "unchanged" the logger's own attributes cannot show.

        ``set_name`` writes to logging's process-wide handler-name registry and
        ``close`` deletes whatever that name points at. A version of this
        function that named the new handler before the swap left the registry
        empty while the handler stayed attached, so the output tests all passed
        and ``getHandlerByName`` returned None.
        """
        if not hasattr(logging, "getHandlerByName"):  # added in 3.12
            self.skipTest("getHandlerByName needs Python 3.12")
        self.assertIs(logging.getHandlerByName(CONSOLE_HANDLER_NAME), handler)

    def test_the_installed_handler_stays_findable_by_name(self):
        """Two calls in a row, which is what an entry point plus a notebook does.

        The second call has to end with the registry pointing at the handler
        that is actually attached. It did not: the new handler took the name
        first, then closing the old one deleted the new one's entry.
        """
        configure_console_logging(stream=self.stream)
        configure_console_logging(stream=self.stream)

        attached = [
            handler
            for handler in self.package_logger.handlers
            if handler.get_name() == CONSOLE_HANDLER_NAME
        ]
        self.assertEqual(len(attached), 1)
        self.assertRegistryStillPointsAt(attached[0])

    def test_notset_leaves_the_ancestors_deciding(self):
        """The one level whose name promises the opposite of what it does.

        On the handler NOTSET means handle everything, but the logger gate comes
        first and NOTSET there means ask the ancestors, whose default is
        WARNING. So INFO is dropped. Pinned rather than fixed: see the docstring
        for why this function does not special-case a single level.
        """
        root = logging.getLogger()
        self.addCleanup(root.setLevel, root.level)
        root.setLevel(logging.WARNING)

        configure_console_logging(level=logging.NOTSET, stream=self.stream)

        self.package_logger.info("Total reward: 7")
        self.assertEqual(self.stream.getvalue(), "")

        # And the documented way to get everything.
        configure_console_logging(level=logging.DEBUG, stream=self.stream)
        self.package_logger.info("Total reward: 7")
        self.assertEqual(self.stream.getvalue(), "Total reward: 7\n")

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
        # Exactly the library default: a NullHandler, which emits nothing.
        # Anything else would put records on a stream nobody asked for, and
        # nothing at all would let a future WARNING reach logging's last-resort
        # handler and land on stderr.
        self.assertEqual(state["handlers"], ["NullHandler"])
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

    def test_running_the_file_directly_enters_main(self):
        """The `if __name__ == "__main__"` block, which calling main() skips.

        Deleting it leaves every other test here green while `python
        evaluate.py config.yaml` does nothing at all. Passing no argument makes
        main() raise before any simulation starts, so this costs nothing: the
        message proves the script reached it, and a script that never calls
        main() exits cleanly with no output.
        """
        completed = subprocess.run(
            [sys.executable, "BalloonPoppingGymEnv/evaluation/evaluate.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout[-500:])
        self.assertIn("Configuration file path is required", completed.stderr)

    def test_the_submission_is_packed_when_the_config_asks_for_it(self):
        """The branch the shipped config actually takes.

        Every other test here sets leaderboard_submission to false, so removing
        the call, or handing it the wrong environment, passed. The config in the
        repository sets it to true.
        """
        from BalloonPoppingGymEnv.evaluation import evaluate

        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        config["leaderboard_submission"] = True
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        env, agent, parameters = object(), object(), {"scenario": {"number": 0}}
        with mock.patch.object(evaluate, "_load_agent_class", return_value=object):
            with mock.patch.object(
                evaluate, "evaluate_scenario", return_value=(env, agent, parameters)
            ):
                with mock.patch(
                    "BalloonPoppingGymEnv.evaluation.results.utils.pack_for_submission"
                ) as packed:
                    evaluate.main([str(self.config_path)])

        packed.assert_called_once()
        self.assertIs(packed.call_args.kwargs["env"], env)
        self.assertIs(packed.call_args.kwargs["scenario_parameters"], parameters)
        self.assertEqual(
            packed.call_args.kwargs["eval_cfg"]["team_name"], config["team_name"]
        )

    def test_the_submission_is_not_packed_when_the_config_declines(self):
        from BalloonPoppingGymEnv.evaluation import evaluate

        with mock.patch.object(evaluate, "_load_agent_class", return_value=object):
            with mock.patch.object(
                evaluate, "evaluate_scenario", return_value=(object(), object(), {})
            ):
                with mock.patch(
                    "BalloonPoppingGymEnv.evaluation.results.utils.pack_for_submission"
                ) as packed:
                    evaluate.main([str(self.config_path)])

        packed.assert_not_called()

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
        """Execute the cell against stubs, rather than read its AST.

        ``ast.walk`` proves both call nodes exist and reports an order, but it
        cannot see reachability: a call under ``if False:`` or inside a nested
        function that is never called satisfies it. One such mutation was caught
        here only because walking happened to reorder the two names, which is
        not something to rely on.
        """
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "evaluate_scenario(" in "".join(cell["source"])
        )

        order = []
        namespace = {
            "configure_console_logging": lambda *a, **k: order.append("configure"),
            "evaluate_scenario": lambda *a, **k: (
                order.append("evaluate") or (object(), object(), {})
            ),
            "NoActionAgent": object,
        }
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

        self.assertEqual(order, ["configure", "evaluate"])

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


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestTheRecordsProductionActuallyEmits(unittest.TestCase):
    """The four calls #51 introduced, not a stand-in that emits them for us.

    The CLI test replaces evaluate_scenario with a fake that logs the two lines
    itself, so turning either real logger.info back into a print stays green
    there. These drive the real functions.

    Each also watches the module's own ``print``. Reading the log records alone
    catches a logger call turned into a print and misses a print *added beside*
    one, which is the same contract broken in the other direction and shows the
    competitor every line twice. Measured: both mutations passed before this.
    Patched per module rather than by capturing stdout, so a print inside
    RocketPy is not counted against us.
    """

    def test_evaluate_scenario_logs_the_completion_and_the_score(self):
        from BalloonPoppingGymEnv.evaluation import evaluate

        env = mock.MagicMock()
        env.reset.return_value = ({}, {})
        env.step.return_value = ({}, 0, True, False, {"popped_count": 7})
        env.trajectories = []

        with mock.patch.object(
            evaluate,
            "load_scenario_parameters",
            return_value=({"scenario": {"random_seed": 0}}, {}),
        ):
            with mock.patch.object(evaluate, "BalloonPoppingEnv", return_value=env):
                with mock.patch.object(evaluate, "save_trajectories"):
                    with mock.patch.object(evaluate, "print", create=True) as printed:
                        with self.assertLogs(
                            "BalloonPoppingGymEnv.evaluation.evaluate", level="INFO"
                        ) as caught:
                            evaluate.evaluate_scenario(
                                lambda *a, **k: mock.MagicMock(),
                                agent_name="A",
                                scenario_number=0,
                                render_mode=None,
                            )

        self.assertEqual(
            [record.getMessage() for record in caught.records],
            [
                "Scenario 0 evaluation completed with agent 'A'.",
                "Total reward: 7",
            ],
        )
        printed.assert_not_called()
        for record in caught.records:
            self.assertEqual(record.levelno, logging.INFO)

    def _run_to_the_end(self, env, action):
        """Step until the episode ends, bounded.

        The bound is not decoration. Both callers below drive a real episode,
        and a termination regression turns an unbounded ``while`` into a run
        that ends at the CI timeout with no useful failure. ``num_timesteps + 5``
        is the same bound ``test_episode_lifecycle`` uses.
        """
        steps = 0
        while True:
            _obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if terminated or truncated:
                return steps
            self.assertLess(steps, env.num_timesteps + 5, "episode did not terminate")

    def test_the_environment_logs_a_timeout(self):
        """The no-launch path, which is also covered in test_episode_lifecycle.

        Deliberately not folded into that test. It is about the lifecycle and
        should not start depending on a logging contract, and this file should
        not start importing its helpers. The duplication costs 0.1 s, because an
        unlaunched step never runs the flight solver: it indexes the balloon
        array that reset() precomputed. Measured: 9999 steps in 0.106 s.
        """
        import numpy as np

        from BalloonPoppingGymEnv.envs import balloon_world
        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        parameters, _ = load_scenario_parameters(0)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        env.reset(seed=parameters["scenario"]["random_seed"])
        action = env.action_space.sample()
        action["launch"] = np.array(0, dtype=action["launch"].dtype)
        for key in ("tvc", "throttle", "roll"):
            action[key] = np.zeros_like(action[key])

        with mock.patch.object(balloon_world, "print", create=True) as printed:
            with self.assertLogs(
                "BalloonPoppingGymEnv.envs.balloon_world", level="INFO"
            ) as caught:
                steps = self._run_to_the_end(env, action)
        printed.assert_not_called()

        self.assertEqual(steps, env.num_timesteps - 1)
        self.assertEqual(
            [record.getMessage() for record in caught.records],
            ["Truncated: Reached max time"],
        )
        for record in caught.records:
            self.assertEqual(record.levelno, logging.INFO)

    def test_the_environment_logs_a_flight_that_finished(self):
        """The fourth call, and the only one that needs a real flight.

        The other branch of the same ``if`` is the timeout above, and reaching
        this one means the rocket has to actually fly and come down inside the
        horizon rather than run out of it. So this launches and steps the
        solver, which is why it is the slow test in this file: around 5800 steps
        and 9 s. Worth it, because with this call back as a bare ``print`` the
        whole suite stayed green.
        """
        import numpy as np

        from BalloonPoppingGymEnv.envs import balloon_world
        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
        from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

        parameters, _ = load_scenario_parameters(0)
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        env.reset(seed=parameters["scenario"]["random_seed"])
        action = env.action_space.sample()
        action["launch"] = np.array(1, dtype=action["launch"].dtype)
        action["launch_inclination_heading"] = np.array([90.0, 0.0], dtype=np.float64)
        action["throttle"] = np.ones_like(action["throttle"])
        for key in ("tvc", "roll"):
            action[key] = np.zeros_like(action[key])

        with mock.patch.object(balloon_world, "print", create=True) as printed:
            with self.assertLogs(
                "BalloonPoppingGymEnv.envs.balloon_world", level="INFO"
            ) as caught:
                steps = self._run_to_the_end(env, action)
        printed.assert_not_called()

        # Ends because the flight ended, not because it ran out of horizon. A
        # timeout would log the other message and this assertion is what keeps
        # the two branches from standing in for each other.
        self.assertLess(steps, env.num_timesteps - 1)
        self.assertEqual(
            [record.getMessage() for record in caught.records],
            ["Terminated: Rocket flight finished"],
        )
        for record in caught.records:
            self.assertEqual(record.levelno, logging.INFO)


if __name__ == "__main__":
    unittest.main()
