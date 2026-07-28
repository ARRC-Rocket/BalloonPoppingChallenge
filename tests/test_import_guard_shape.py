"""No test may treat a broken simulation stack as an absent one.

``import rocketpy`` runs the package's own ``__init__``, which imports its
submodules. If one of those raises ``ImportError``, wrapping the import in
``try / except ImportError`` catches it and the file reports itself as "stack not
installed". The suite then goes green while skipping the very tests that exist to
notice an ActiveRocketPy update breaking the import.

``find_spec("rocketpy") is not None`` separates the two: absent means skip,
present but broken means the import fails at collection.

This reads the source with ``ast`` and never imports it, so it runs anywhere, in
the same spirit as the other invariant tests here.
"""

import ast
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Guarding these would hide the same class of failure: a stack that is installed
# but no longer imports, or a package symbol that has been renamed.
GUARDED_NAMES_THAT_MUST_NOT_BE = frozenset(
    {"rocketpy", "BalloonPoppingGymEnv", "numpy", "yaml"}
)


def _imported_roots(node):
    """Top-level package names imported anywhere inside ``node``."""
    roots = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in child.names)
        elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
            roots.add(child.module.split(".")[0])
    return roots


def _catches_import_error(handler):
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for name in names:
        if isinstance(name, ast.Name) and name.id in ("ImportError", "Exception"):
            return True
    return False


class TestNoTestGuardsAnImportBehindImportError(unittest.TestCase):
    def test_no_test_file_hides_a_broken_stack_as_a_skip(self):
        offenders = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            if path.name == Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                if not any(_catches_import_error(h) for h in node.handlers):
                    continue
                guarded = _imported_roots(node) & GUARDED_NAMES_THAT_MUST_NOT_BE
                if guarded:
                    offenders.append(
                        f"{path.name}:{node.lineno} guards {sorted(guarded)}"
                    )

        self.assertEqual(
            offenders,
            [],
            "use find_spec for the absent case and import outside the guard:\n"
            + "\n".join(offenders),
        )

    def test_the_check_can_actually_see_a_guarded_import(self):
        # Guards the guard: the walk above has to find an import nested inside a
        # try block, or the whole test passes for the wrong reason.
        tree = ast.parse("try:\n    import rocketpy\nexcept ImportError:\n    pass\n")
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try))

        self.assertTrue(any(_catches_import_error(h) for h in node.handlers))
        self.assertIn("rocketpy", _imported_roots(node))


if __name__ == "__main__":
    unittest.main()
