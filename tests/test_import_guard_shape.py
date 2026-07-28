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
import builtins
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def _catches_import_failure(handler):
    """Whether this handler would swallow an ``ImportError``.

    Derived from the real exception hierarchy rather than a list of names. A
    hand-kept list missed ``ModuleNotFoundError``, which is the exception a
    package raises when one of its own submodules is absent, and therefore the
    exact one this file exists to notice.
    """
    if handler.type is None:
        return True

    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for node in names:
        if not isinstance(node, ast.Name):
            continue
        caught = getattr(builtins, node.id, None)
        if not isinstance(caught, type) or not issubclass(caught, BaseException):
            continue
        # Either it is an import failure, like ImportError or
        # ModuleNotFoundError, or it is broad enough to contain one.
        if issubclass(caught, ImportError) or issubclass(ImportError, caught):
            return True
    return False


def _imports_in_try_body(node):
    """Import statements in the ``try`` body only.

    ``ast.walk`` over the whole ``Try`` also reaches the handlers, ``else`` and
    ``finally``. An import in ``else`` runs after the handler can no longer fire,
    so reporting it would be a false positive.
    """
    found = []
    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append(child)
    return found


def _described(import_node):
    if isinstance(import_node, ast.Import):
        return ", ".join(alias.name for alias in import_node.names)
    module = import_node.module or "."
    return f"from {module}"


class TestNoTestGuardsAnImportBehindImportError(unittest.TestCase):
    def test_no_test_file_hides_a_broken_stack_as_a_skip(self):
        """Any guarded import, not a list of names.

        A denylist of four packages let the same failure return through any other
        dependency. Nothing in this suite has a reason to import inside an
        import-catching try, so the rule is simply that none may.
        """
        offenders = []
        for path in sorted(TESTS_DIR.rglob("test_*.py")):
            if path.name == Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                if not any(_catches_import_failure(h) for h in node.handlers):
                    continue
                for guarded in _imports_in_try_body(node):
                    offenders.append(
                        f"{path.name}:{guarded.lineno} guards {_described(guarded)}"
                    )

        self.assertEqual(
            offenders,
            [],
            "use find_spec for the absent case and import outside the guard:\n"
            + "\n".join(offenders),
        )

    def test_every_import_failure_exception_is_recognised(self):
        # ModuleNotFoundError is the one a hand-kept list missed, and it is what
        # a package raises for its own missing submodule.
        for name in (
            "ImportError",
            "ModuleNotFoundError",
            "Exception",
            "BaseException",
        ):
            with self.subTest(exception=name):
                tree = ast.parse(f"try:\n    import x\nexcept {name}:\n    pass\n")
                node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try))
                self.assertTrue(_catches_import_failure(node.handlers[0]))

    def test_a_bare_except_and_a_tuple_both_count(self):
        for source in (
            "try:\n    import x\nexcept:\n    pass\n",
            "try:\n    import x\nexcept (OSError, ModuleNotFoundError):\n    pass\n",
        ):
            with self.subTest(source=source.splitlines()[2]):
                tree = ast.parse(source)
                node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try))
                self.assertTrue(_catches_import_failure(node.handlers[0]))

    def test_an_unrelated_exception_is_not_reported(self):
        tree = ast.parse("try:\n    import x\nexcept ValueError:\n    pass\n")
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try))

        self.assertFalse(_catches_import_failure(node.handlers[0]))

    def test_an_import_in_the_else_block_is_not_reported(self):
        # It runs after the handler can no longer fire, so the handler cannot
        # hide its failure. Walking the whole Try node reported it as guarded.
        tree = ast.parse(
            "try:\n    import a\nexcept ImportError:\n    pass\n"
            "else:\n    import rocketpy\n"
        )
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try))

        guarded = [_described(i) for i in _imports_in_try_body(node)]
        self.assertEqual(guarded, ["a"])

    def test_the_check_can_actually_see_a_guarded_import(self):
        tree = ast.parse("try:\n    import rocketpy\nexcept ImportError:\n    pass\n")
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try))

        self.assertTrue(_catches_import_failure(node.handlers[0]))
        self.assertEqual(
            [_described(i) for i in _imports_in_try_body(node)], ["rocketpy"]
        )


if __name__ == "__main__":
    unittest.main()
