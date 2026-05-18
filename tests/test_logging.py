"""Logging-hygiene invariant for the BalloonPoppingGymEnv package (issue #14).

Diagnostic output must go through the `logging` module, not bare `print()`,
so verbosity is controllable: quiet during scoring or CI, verbose when
debugging. AST-only check -- parses the package sources and runs without
rocketpy or any heavyweight dependency.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "BalloonPoppingGymEnv"


def _iter_package_python_files():
    """Yield every .py file inside the BalloonPoppingGymEnv package."""
    yield from PACKAGE_ROOT.rglob("*.py")


class TestNoBarePrint(unittest.TestCase):
    """Production sources must route diagnostics through `logging`, not `print()`."""

    def test_no_print_calls_in_package(self):
        offenders = []
        for path in _iter_package_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            f"bare print() in package sources; use a module logger instead: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
