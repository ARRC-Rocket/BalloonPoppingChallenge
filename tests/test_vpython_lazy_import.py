"""vpython may be imported only by the branch that renders with it.

vpython is an optional, render-only dependency. Importing it anywhere that runs
at import time forces every consumer, including Colab running with
`render_mode=None`, to install it just to import the environment.

Written against the exact node rather than by searching. The version this
replaced walked the whole tree for a function named `_render_frame` and accepted
an import anywhere beneath it, which passed on all of: an import at the top of
the method, one in an unused nested function, one in a decoy function outside
the class, one in a module-level `try`, and `if True: import vpython`. That last
one is the bug this file exists to catch, and importing balloon_world raised
ModuleNotFoundError while both assertions stayed green.

AST-only, so it runs without rocketpy, vpython, or anything heavy.
"""

import ast
import subprocess
import sys
import textwrap
import unittest
from importlib.util import find_spec
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BALLOON_WORLD = REPO_ROOT / "BalloonPoppingGymEnv" / "envs" / "balloon_world.py"

ENVIRONMENT_CLASS = "BalloonPoppingEnv"
RENDER_METHOD = "_render_frame"
RENDER_MODE = "vpython"


def _imports_vpython(node):
    """Whether this statement imports the vpython package."""
    if isinstance(node, ast.Import):
        return any(alias.name.split(".", 1)[0] == "vpython" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".", 1)[0] == "vpython"
    return False


def _dynamic_vpython_import(node):
    """Whether this expression imports vpython without an import statement.

    `__import__("vpython")` and `importlib.import_module("vpython")` reach the
    same machinery and are invisible to the check above.
    """
    if not isinstance(node, ast.Call):
        return False
    target = node.func
    if isinstance(target, ast.Name):
        name = target.id
    elif isinstance(target, ast.Attribute):
        name = target.attr
    else:
        return False
    if name not in ("__import__", "import_module"):
        return False
    # Keywords as well as positionals: `import_module(name="vpython")` is the
    # same call, and reading only `Call.args` let it through.
    arguments = list(node.args) + [keyword.value for keyword in node.keywords]
    return any(
        isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.split(".", 1)[0] == "vpython"
        for argument in arguments
    )


def _reaches_vpython(node):
    return _imports_vpython(node) or _dynamic_vpython_import(node)


def _is_render_mode(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "render_mode"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_the_literal(node):
    return isinstance(node, ast.Constant) and node.value == RENDER_MODE


def _guards_the_vpython_branch(node):
    """Whether this test is `self.render_mode == "vpython"`, either way round."""
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
    ):
        return False
    left, right = node.left, node.comparators[0]
    return (_is_render_mode(left) and _is_the_literal(right)) or (
        _is_the_literal(left) and _is_render_mode(right)
    )


def _exactly_one(candidates, description):
    found = list(candidates)
    if len(found) != 1:
        raise AssertionError(f"expected exactly one {description}, found {len(found)}")
    return found[0]


def _render_frame_of(tree):
    """`BalloonPoppingEnv._render_frame`, by position rather than by search."""
    environment = _exactly_one(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == ENVIRONMENT_CLASS
        ),
        f"{ENVIRONMENT_CLASS} class at module level",
    )
    return _exactly_one(
        (
            node
            for node in environment.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == RENDER_METHOD
        ),
        f"{ENVIRONMENT_CLASS}.{RENDER_METHOD} method",
    )


def _branch_imports_of(render_frame):
    """Imports that are direct statements of the vpython branch.

    Direct, not descendant: an import inside a nested function under the branch
    does not run when the branch does.
    """
    branch = _exactly_one(
        (
            statement
            for statement in render_frame.body
            if isinstance(statement, ast.If)
            and _guards_the_vpython_branch(statement.test)
        ),
        f'`self.render_mode == "{RENDER_MODE}"` branch of {RENDER_METHOD}',
    )
    return [statement for statement in branch.body if _imports_vpython(statement)]


def _stray_imports(source):
    """vpython imports that are not the render branch's own, by identity.

    A copy elsewhere in the tree is a different node, so it is reported however
    it is spelled and wherever it sits.
    """
    tree = ast.parse(source)
    allowed = {id(node) for node in _branch_imports_of(_render_frame_of(tree))}
    return [
        node
        for node in ast.walk(tree)
        if _reaches_vpython(node) and id(node) not in allowed
    ]


class TestVpythonIsImportedOnlyWhereItIsUsed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BALLOON_WORLD.read_text(encoding="utf-8")

    def test_the_render_branch_imports_it(self):
        imports = _branch_imports_of(_render_frame_of(ast.parse(self.source)))

        self.assertTrue(
            imports,
            f'the `self.render_mode == "{RENDER_MODE}"` branch must import vpython',
        )

    def test_nothing_else_in_the_file_imports_it(self):
        stray = _stray_imports(self.source)

        self.assertEqual(
            [f"line {node.lineno}" for node in stray],
            [],
            "vpython may only be imported by the render branch itself",
        )


class TestTheModuleImportsWithVpythonBlocked(unittest.TestCase):
    """The behaviour the AST check is a proxy for.

    An alias defeats the source check without much trouble
    (`from importlib import import_module as load`), and chasing that needs name
    binding analysis. This asks the question directly instead: with vpython
    refused by the import system, does the package still import?

    In a subprocess because the answer depends on what is already in
    `sys.modules`, and this one is only meaningful from a clean interpreter.
    """

    def test_importing_the_environment_does_not_need_vpython(self):
        if find_spec("rocketpy") is None:
            self.skipTest("simulation stack not installed")

        program = textwrap.dedent("""
            import sys

            class Refuse:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".", 1)[0] == "vpython":
                        raise ImportError("vpython is not available here")
                    return None

            sys.meta_path.insert(0, Refuse())
            import BalloonPoppingGymEnv.envs.balloon_world  # noqa: F401
            print("imported")
        """)
        finished = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        self.assertEqual(
            finished.returncode,
            0,
            f"importing the environment reached vpython:\n{finished.stderr[-2000:]}",
        )
        self.assertIn("imported", finished.stdout)


class TestTheCheckerCatchesWhatItIsFor(unittest.TestCase):
    """Every shape here passed the version this file replaced.

    Run against source built here, so the check is exercised by what it has to
    reject without anyone breaking the environment to find out.
    """

    _GOOD = """
class BalloonPoppingEnv:
    def _render_frame(self):
        if self.render_mode == "vpython":
            from vpython import canvas

            return canvas()
        return None
"""

    def _accepted(self, source):
        """Whether both checks pass on this source."""
        try:
            if not _branch_imports_of(_render_frame_of(ast.parse(source))):
                return False
            return not _stray_imports(source)
        except AssertionError:
            return False

    def test_the_real_shape_passes(self):
        self.assertTrue(self._accepted(self._GOOD))

    def test_the_real_file_passes(self):
        """The control: the checks above mean nothing if nothing passes them."""
        self.assertTrue(self._accepted(BALLOON_WORLD.read_text(encoding="utf-8")))

    def test_an_import_that_runs_at_import_time_fails(self):
        """`if True:` is the one that actually breaks the package import."""
        for prefix in (
            "import vpython\n",
            "if True:\n    import vpython\n",
            "try:\n    import vpython\nexcept ImportError:\n    vpython = None\n",
            'vpython = __import__("vpython")\n',
            'import importlib\n\nvpython = importlib.import_module("vpython")\n',
        ):
            with self.subTest(prefix=prefix.splitlines()[0]):
                self.assertFalse(self._accepted(prefix + self._GOOD))

    def test_an_import_at_the_top_of_the_method_fails(self):
        """It would run on every call, including the matplotlib one."""
        source = self._GOOD.replace(
            "    def _render_frame(self):\n",
            "    def _render_frame(self):\n        import vpython\n",
        )

        self.assertFalse(self._accepted(source))

    def test_an_import_in_a_nested_function_fails(self):
        """Nested means it does not run when the branch does."""
        source = self._GOOD.replace(
            "            from vpython import canvas\n",
            "            def _unused():\n                from vpython import canvas\n",
        )

        self.assertFalse(self._accepted(source))

    def test_an_import_in_the_class_body_fails(self):
        source = self._GOOD.replace(
            "class BalloonPoppingEnv:\n",
            "class BalloonPoppingEnv:\n    import vpython\n",
        )

        self.assertFalse(self._accepted(source))

    def test_a_decoy_function_of_the_same_name_fails(self):
        """The version this replaced took the first match by name, anywhere."""
        source = (
            "def _render_frame(self):\n    from vpython import canvas\n\n\n"
            + self._GOOD.replace(
                "            from vpython import canvas\n", "            pass\n"
            )
        )

        self.assertFalse(self._accepted(source))

    def test_an_import_in_another_branch_fails(self):
        source = self._GOOD.replace(
            "        return None\n",
            '        if self.render_mode == "matplotlib":\n'
            "            from vpython import canvas\n"
            "        return None\n",
        )

        self.assertFalse(self._accepted(source))

    def test_a_missing_branch_fails_rather_than_passing_vacuously(self):
        source = self._GOOD.replace(
            '        if self.render_mode == "vpython":\n'
            "            from vpython import canvas\n\n"
            "            return canvas()\n",
            "        pass\n",
        )

        self.assertFalse(self._accepted(source))


if __name__ == "__main__":
    unittest.main()
