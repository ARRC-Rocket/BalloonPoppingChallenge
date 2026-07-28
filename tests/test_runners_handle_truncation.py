"""Every loop that drives the environment has to handle both end conditions.

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

What is checked, and why not something cheaper
----------------------------------------------
The first version of this file asked whether both names appeared somewhere in
the loop condition, and whether something called ``truncated`` was assigned. Run
against hand written snippets, that let all of these through:

    while not (terminated and truncated):     # stops only if both, so never
    while not terminated or truncated:        # precedence: keeps going on truncated
    while terminated or truncated:            # inverted, body never runs
    o, r, _, truncated, i = env.step(action)  # terminated discarded, never updated
    o, r, truncated, terminated, i = ...      # the two swapped
    truncated = env.step(action)              # the whole tuple, truthy after one pass

The first two keep calling ``step()`` after a flag is set, which is the failure
this file exists to prevent, so name membership was not a check.

So the guard is evaluated over all four combinations of the two flags and has to
agree with ``not (terminated or truncated)``, and the unpack has to be five
elements with the flags in positions 2 and 3. Evaluating the condition is safe
by construction: it is refused first unless the only names in it are the two
flags, and it runs with no builtins.

AST only, so it runs without the simulation stack.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Walked directly instead of filtering the whole tree afterwards. rglob from the
# repository root opens ``.venv`` and the ActiveRocketPy submodule before
# anything gets a chance to exclude them, and three test methods would each pay
# for it.
SOURCE_ROOTS = ("BalloonPoppingGymEnv", "doc", "scripts")

# The loops that exist today. A subset check rather than equality: adding a
# runner is fine, losing one silently is not. ``len(found) >= 3`` was not this,
# since one runner disappearing while an unrelated ``.step()`` loop appeared
# left the count unchanged.
KNOWN_RUNNERS = {
    Path("BalloonPoppingGymEnv/evaluation/evaluate.py"),
    Path("doc/examples/run_env_agent.py"),
    Path("doc/examples/test_navigation_agent.py"),
}

STEP_RETURN_ARITY = 5
TERMINATED_INDEX = 2
TRUNCATED_INDEX = 3
FLAG_NAMES = ("terminated", "truncated")

# Keep going only while neither flag is set. Keys are (terminated, truncated).
KEEP_GOING = {
    (False, False): True,
    (False, True): False,
    (True, False): False,
    (True, True): False,
}


def _is_env_step_call(node):
    """A call to ``.step()`` on something named for the environment.

    The receiver is looked at so that a future ``optimizer.step()`` or
    ``scheduler.step()`` inside a ``while`` is not conscripted into being an
    episode loop and asked for flags it has no reason to have.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
    ):
        return False
    receiver = node.func.value
    if isinstance(receiver, ast.Name):
        name = receiver.id
    elif isinstance(receiver, ast.Attribute):
        name = receiver.attr
    else:
        return False
    return "env" in name.lower()


def _step_assignments(node):
    """Assignments in this subtree whose value is a direct environment step."""
    return [
        inner
        for inner in ast.walk(node)
        if isinstance(inner, ast.Assign) and _is_env_step_call(inner.value)
    ]


def episode_loops(tree):
    """Every ``while`` whose body assigns from an environment step."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.While) and _step_assignments(node)
    ]


def guard_problem(loop):
    """What is wrong with this loop's condition, or None.

    Judged by what it does rather than by what it says. ``not (terminated and
    truncated)`` mentions both names and keeps stepping after either one fires.
    """
    names = {inner.id for inner in ast.walk(loop.test) if isinstance(inner, ast.Name)}
    if not names:
        return "the condition never looks at either flag"
    if not names <= set(FLAG_NAMES):
        others = sorted(names - set(FLAG_NAMES))
        return f"the condition also depends on {others}, which this cannot evaluate"
    if names != set(FLAG_NAMES):
        return f"the condition only looks at {sorted(names)}"

    expression = ast.Expression(body=loop.test)
    ast.fix_missing_locations(expression)
    code = compile(expression, "<loop condition>", "eval")
    behaviour = {
        (terminated, truncated): bool(
            eval(  # noqa: S307 - only the two flag names, and no builtins
                code,
                {"__builtins__": {}},
                {"terminated": terminated, "truncated": truncated},
            )
        )
        for terminated in (False, True)
        for truncated in (False, True)
    }
    if behaviour != KEEP_GOING:
        keeps_going = sorted(state for state, goes in behaviour.items() if goes)
        return (
            f"the condition keeps stepping for (terminated, truncated) in "
            f"{keeps_going}; it has to keep going only for (False, False)"
        )
    return None


def unpack_problem(loop):
    """What is wrong with how this loop takes ``step()`` apart, or None."""
    for assignment in _step_assignments(loop):
        if len(assignment.targets) != 1:
            return "the step result is assigned to more than one target"
        target = assignment.targets[0]
        if not isinstance(target, (ast.Tuple, ast.List)):
            return "the whole step result is assigned to a single name"
        if len(target.elts) != STEP_RETURN_ARITY:
            return (
                f"the step result is unpacked into {len(target.elts)} names, "
                f"and step() returns {STEP_RETURN_ARITY}"
            )
        for index, expected in (
            (TERMINATED_INDEX, "terminated"),
            (TRUNCATED_INDEX, "truncated"),
        ):
            element = target.elts[index]
            got = element.id if isinstance(element, ast.Name) else ast.dump(element)
            if got != expected:
                return f"position {index} of the step result is {got}, not {expected}"
    return None


class TestTheRunnersInThisRepository(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loops = []
        for root in SOURCE_ROOTS:
            directory = REPO_ROOT / root
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*.py")):
                # Deliberately not catching SyntaxError. doc/examples is outside
                # the ruff paths CI checks, so a broken example would otherwise
                # drop out of discovery while everything stayed green.
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for loop in episode_loops(tree):
                    cls.loops.append((path.relative_to(REPO_ROOT), loop))

    def test_the_known_runners_were_all_found(self):
        """Or every assertion below holds over whatever is left.

        The discovery is the part that can quietly stop working: a rename, a
        move, or narrowing what counts as a step call.
        """
        found = {path for path, _ in self.loops}

        self.assertLessEqual(
            KNOWN_RUNNERS,
            found,
            f"these runners were not discovered: {sorted(KNOWN_RUNNERS - found)}",
        )

    def test_every_loop_stops_when_either_flag_is_set(self):
        for path, loop in self.loops:
            with self.subTest(runner=f"{path}:{loop.lineno}"):
                self.assertIsNone(guard_problem(loop), f"{path} line {loop.lineno}")

    def test_every_loop_takes_both_flags_out_of_the_step_result(self):
        for path, loop in self.loops:
            with self.subTest(runner=f"{path}:{loop.lineno}"):
                self.assertIsNone(unpack_problem(loop), f"{path} line {loop.lineno}")


class TestTheCheckerItself(unittest.TestCase):
    """The checker is the thing that can be wrong without anything noticing.

    Its only other input is three runners that already pass, and that cannot
    tell a working check from one that returns None for everything.
    """

    GOOD = "while not (terminated or truncated):\n    o, r, terminated, truncated, i = env.step(a)"

    @staticmethod
    def _only_loop(source):
        loops = episode_loops(ast.parse(source))
        assert len(loops) == 1, f"expected one episode loop, found {len(loops)}"
        return loops[0]

    def test_the_correct_loop_has_no_problem(self):
        loop = self._only_loop(self.GOOD)

        self.assertIsNone(guard_problem(loop))
        self.assertIsNone(unpack_problem(loop))

    def test_the_flags_may_be_written_in_either_order(self):
        loop = self._only_loop(
            "while not (truncated or terminated):\n"
            "    o, r, terminated, truncated, i = env.step(a)"
        )

        self.assertIsNone(guard_problem(loop))

    def test_a_guard_that_needs_both_flags_is_rejected(self):
        loop = self._only_loop(self.GOOD.replace("or truncated", "and truncated"))

        self.assertIsNotNone(guard_problem(loop))

    def test_a_guard_with_the_wrong_precedence_is_rejected(self):
        loop = self._only_loop(
            self.GOOD.replace(
                "not (terminated or truncated)", "not terminated or truncated"
            )
        )

        self.assertIsNotNone(guard_problem(loop))

    def test_an_inverted_guard_is_rejected(self):
        loop = self._only_loop(
            self.GOOD.replace(
                "not (terminated or truncated)", "terminated or truncated"
            )
        )

        self.assertIsNotNone(guard_problem(loop))

    def test_a_guard_that_only_looks_at_one_flag_is_rejected(self):
        loop = self._only_loop(
            self.GOOD.replace("not (terminated or truncated)", "not terminated")
        )

        self.assertIsNotNone(guard_problem(loop))

    def test_a_guard_driven_by_something_else_is_rejected(self):
        loop = self._only_loop(
            self.GOOD.replace("not (terminated or truncated)", "running")
        )

        self.assertIsNotNone(guard_problem(loop))

    def test_discarding_terminated_is_rejected(self):
        loop = self._only_loop(
            self.GOOD.replace("r, terminated, truncated", "r, _, truncated")
        )

        self.assertIsNotNone(unpack_problem(loop))

    def test_discarding_truncated_is_rejected(self):
        loop = self._only_loop(
            self.GOOD.replace("terminated, truncated, i", "terminated, _, i")
        )

        self.assertIsNotNone(unpack_problem(loop))

    def test_swapping_the_two_flags_is_rejected(self):
        loop = self._only_loop(
            self.GOOD.replace("terminated, truncated, i", "truncated, terminated, i")
        )

        self.assertIsNotNone(unpack_problem(loop))

    def test_assigning_the_whole_result_to_one_name_is_rejected(self):
        loop = self._only_loop(
            "while not (terminated or truncated):\n    truncated = env.step(a)"
        )

        self.assertIsNotNone(unpack_problem(loop))

    def test_an_unpack_of_the_wrong_length_is_rejected(self):
        loop = self._only_loop(
            "while not (terminated or truncated):\n"
            "    o, terminated, truncated, i = env.step(a)"
        )

        self.assertIsNotNone(unpack_problem(loop))

    def test_an_unrelated_step_call_is_not_an_episode_loop(self):
        """Or adding a training loop would demand flags it has no reason to have."""
        for source in (
            "while training:\n    loss = optimizer.step()",
            "while training:\n    scheduler.step()",
        ):
            with self.subTest(source=source):
                self.assertEqual(episode_loops(ast.parse(source)), [])

    def test_a_step_call_that_is_not_assigned_is_not_an_episode_loop(self):
        self.assertEqual(episode_loops(ast.parse("while x:\n    env.step(a)")), [])


if __name__ == "__main__":
    unittest.main()
