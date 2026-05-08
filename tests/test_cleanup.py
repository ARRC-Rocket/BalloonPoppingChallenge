"""Cleanup invariants for BalloonPoppingGymEnv.

Static checks on the production sources that catch dead code, redundant
calls, drifted example configs, and empty f-strings.  AST-only -- runs
without rocketpy or any heavyweight dependency.
"""

import ast
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "BalloonPoppingGymEnv"


def _iter_package_python_files():
    """Yield every .py file inside the BalloonPoppingGymEnv package."""
    yield from PACKAGE_ROOT.rglob("*.py")


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestRedundantFileClose(unittest.TestCase):
    """D1: `with open(...) as f:` followed by `f.close()` is redundant."""

    def test_no_redundant_close_after_with(self):
        offenders = []
        for path in _iter_package_python_files():
            tree = _parse(path)
            for parent in ast.walk(tree):
                for body_attr in ("body", "orelse", "finalbody"):
                    body = getattr(parent, body_attr, None)
                    if not isinstance(body, list):
                        continue
                    for i in range(len(body) - 1):
                        with_stmt = body[i]
                        if not isinstance(with_stmt, ast.With):
                            continue
                        bound = {
                            item.optional_vars.id
                            for item in with_stmt.items
                            if isinstance(item.optional_vars, ast.Name)
                        }
                        nxt = body[i + 1]
                        if (
                            isinstance(nxt, ast.Expr)
                            and isinstance(nxt.value, ast.Call)
                            and isinstance(nxt.value.func, ast.Attribute)
                            and isinstance(nxt.value.func.value, ast.Name)
                            and nxt.value.func.attr == "close"
                            and nxt.value.func.value.id in bound
                        ):
                            offenders.append(
                                f"{path.relative_to(REPO_ROOT)}:{nxt.lineno}"
                            )
        self.assertEqual(
            offenders,
            [],
            f"redundant `.close()` after `with open(...)`: {offenders}",
        )


class TestNavigationAgentNoClassLevelImport(unittest.TestCase):
    """D2: NavigationAgent class body must not declare imports."""

    def test_class_body_has_no_imports(self):
        path = PACKAGE_ROOT / "agents" / "example_agents.py"
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NavigationAgent":
                offenders = [
                    f"line {n.lineno}"
                    for n in node.body
                    if isinstance(n, (ast.Import, ast.ImportFrom))
                ]
                self.assertEqual(
                    offenders,
                    [],
                    f"NavigationAgent has class-level imports: {offenders}",
                )
                return
        self.fail("NavigationAgent class not found in example_agents.py")


class TestExampleEvalConfigKeysConsistent(unittest.TestCase):
    """D3: every key in example_eval_cfg.yaml's agent_kwargs must be consumed."""

    def test_all_kwargs_keys_are_consumed_by_agent(self):
        cfg_path = PACKAGE_ROOT / "evaluation" / "configs" / "example_eval_cfg.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        agent_kwargs = cfg.get("agent_kwargs") or {}
        agent_module_path = (REPO_ROOT / cfg["agent_module_path"]).resolve()
        agent_class_name = cfg["agent_class_name"]
        consumed = self._consumed_kwargs(_parse(agent_module_path), agent_class_name)
        unused = set(agent_kwargs.keys()) - consumed
        self.assertEqual(
            unused,
            set(),
            f"keys in {cfg_path.name} not consumed by {agent_class_name}: {unused}",
        )

    @staticmethod
    def _consumed_kwargs(tree, class_name):
        """Names passed to `kwargs.get("...", ...)` inside `<class_name>.__init__`."""
        keys = set()
        for cls in ast.walk(tree):
            if not (isinstance(cls, ast.ClassDef) and cls.name == class_name):
                continue
            for fn in cls.body:
                if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                    continue
                for call in ast.walk(fn):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "get"
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "kwargs"
                        and call.args
                        and isinstance(call.args[0], ast.Constant)
                        and isinstance(call.args[0].value, str)
                    ):
                        keys.add(call.args[0].value)
        return keys


class TestNoEmptyFStrings(unittest.TestCase):
    """D4: every f-string must contain at least one interpolated value."""

    def test_all_fstrings_have_interpolation(self):
        offenders = []

        class _Visitor(ast.NodeVisitor):
            def __init__(self, rel_path):
                self.rel_path = rel_path

            def visit_FormattedValue(self, node):
                # format_spec (e.g. the `.2f` in `{x:.2f}`) is itself a JoinedStr
                # without FormattedValue children -- skip it so we only flag
                # source-level f-string literals.
                if node.value is not None:
                    self.visit(node.value)

            def visit_JoinedStr(self, node):
                if not any(isinstance(v, ast.FormattedValue) for v in node.values):
                    offenders.append(f"{self.rel_path}:{node.lineno}")
                self.generic_visit(node)

        for path in _iter_package_python_files():
            _Visitor(path.relative_to(REPO_ROOT)).visit(_parse(path))

        self.assertEqual(
            offenders,
            [],
            f"f-strings without interpolation: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
