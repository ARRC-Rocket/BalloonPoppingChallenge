"""Every loop that drives the environment has to look at both end conditions.

``step()`` reports the flight ending as ``terminated`` and running out of
precomputed horizon as ``truncated``. A loop that waits only for ``terminated``
never stops on the second one: it keeps calling ``step()``, ``current_step``
walks past the end of ``_balloon_flights`` and the run dies indexing it. For
scenario 1 the horizon is the usual ending, so this is the common path rather
than an edge.

The change that split the two causes said it moved both runners with it. There
were three. ``evaluate.py`` and ``run_env_agent.py`` were updated and
``doc/examples/test_navigation_agent.py`` was reformatted and left waiting on
``terminated`` alone. It stayed hidden because scenario 0 lands at step 5896 of
9999, so ``terminated`` fires first there and the loop never reaches the case it
gets wrong.

Discovered rather than listed, so a fourth runner cannot be added and miss this.
AST only, so it runs without the simulation stack.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Anything that is not ours or is not a runner. ``tests`` is excluded because a
# test may legitimately drive the environment in a way that is about something
# else, and it is not what a competitor copies from.
SKIP_DIRECTORIES = {".venv", "ActiveRocketPy", "tests", ".git", "build", "dist"}


def _python_files():
    for path in REPO_ROOT.rglob("*.py"):
        if SKIP_DIRECTORIES.isdisjoint(
            part for part in path.relative_to(REPO_ROOT).parts
        ):
            yield path


def _calls_step(node):
    """Whether this subtree calls ``something.step(...)``."""
    return any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "step"
        for inner in ast.walk(node)
    )


def _episode_loops():
    """Every ``while`` whose body steps the environment, with its file."""
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file is another test's job
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.While) and any(
                _calls_step(statement) for statement in node.body
            ):
                yield path.relative_to(REPO_ROOT), node


class TestEveryRunnerHandlesTruncation(unittest.TestCase):
    def test_at_least_one_episode_loop_was_found(self):
        """Or every assertion below would hold over an empty set.

        The discovery is the part that can silently stop working: a rename, a
        moved directory or a refactor into a helper would leave this file
        passing while checking nothing.
        """
        found = [str(path) for path, _ in _episode_loops()]

        self.assertGreaterEqual(
            len(found), 3, f"expected the three known runners, found {found}"
        )

    def test_every_episode_loop_waits_on_both_flags(self):
        for path, loop in _episode_loops():
            with self.subTest(runner=str(path)):
                names = {
                    inner.id
                    for inner in ast.walk(loop.test)
                    if isinstance(inner, ast.Name)
                }
                self.assertIn(
                    "truncated",
                    names,
                    f"{path} line {loop.lineno} loops on {sorted(names)}. Running "
                    "out of horizon is reported as truncated, so this loop keeps "
                    "stepping past the end of the precomputed flights",
                )
                self.assertIn("terminated", names, f"{path} line {loop.lineno}")

    def test_every_episode_loop_keeps_the_truncated_flag_step_returns(self):
        """Waiting on it is no use if the assignment throws it away.

        ``observation, reward, terminated, _, info = env.step(action)`` reads
        correctly and leaves ``truncated`` at whatever it was initialised to,
        which is False forever.
        """
        for path, loop in _episode_loops():
            with self.subTest(runner=str(path)):
                targets = set()
                for inner in ast.walk(loop):
                    if isinstance(inner, ast.Assign) and _calls_step(inner.value):
                        for target in inner.targets:
                            targets |= {
                                element.id
                                for element in ast.walk(target)
                                if isinstance(element, ast.Name)
                            }
                self.assertIn(
                    "truncated",
                    targets,
                    f"{path} line {loop.lineno} discards the truncated flag that "
                    "step() returns",
                )


if __name__ == "__main__":
    unittest.main()
