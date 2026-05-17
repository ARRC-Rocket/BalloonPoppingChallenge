"""Import-hygiene invariant for vpython in BalloonPoppingGymEnv.

vpython is an optional, render-only dependency.  Importing it at module
scope forces every consumer -- including Colab, which runs with
render_mode=None -- to install vpython just to import the environment.
The import belongs inside the vpython branch of `_render_frame`.

AST-only -- runs without rocketpy, vpython, or any heavyweight dependency.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BALLOON_WORLD = REPO_ROOT / "BalloonPoppingGymEnv" / "envs" / "balloon_world.py"


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports_vpython(node):
    """True if an Import / ImportFrom node pulls in the vpython package."""
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "vpython" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "vpython"
    return False


class TestVpythonLazyImport(unittest.TestCase):
    """vpython must be imported lazily, not at module scope."""

    def test_no_module_level_vpython_import(self):
        tree = _parse(BALLOON_WORLD)
        offenders = [
            f"balloon_world.py:{node.lineno}"
            for node in tree.body
            if _imports_vpython(node)
        ]
        self.assertEqual(
            offenders,
            [],
            f"vpython must not be imported at module scope: {offenders}",
        )

    def test_vpython_imported_inside_render_frame(self):
        tree = _parse(BALLOON_WORLD)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_render_frame":
                lazy = [n for n in ast.walk(node) if _imports_vpython(n)]
                self.assertTrue(
                    lazy,
                    "_render_frame should import vpython lazily inside its body",
                )
                return
        self.fail("_render_frame not found in balloon_world.py")


if __name__ == "__main__":
    unittest.main()
