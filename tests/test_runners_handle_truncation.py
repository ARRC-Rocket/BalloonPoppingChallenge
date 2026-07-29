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

Three ways this went unchecked, each measured: ``"env" in name.lower()`` never
found ``world = gym.make(...)``, ``ast.walk`` gave an outer ``while`` the inner
loop's step, and the scan skipped ``tests/``, where two loops waited on one flag.

AST only, so it runs without the simulation stack.
"""

import ast
import os
import unittest
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything under the repository root except these, pruned during the walk so
# ``.venv`` and the ActiveRocketPy fork are never opened: 46 files in 0.03 s. The
# old ``("BalloonPoppingGymEnv", "doc", "scripts")`` left ``tests/`` unscanned.
EXCLUDED_DIRECTORIES = frozenset(
    {".venv", ".ci-venv", "ActiveRocketPy", "build", "dist", "node_modules"}
)

# The loops that exist today, each pinned to the function it lives in (``None``
# for one written at module level). Adding a runner is fine, losing one silently
# is not.
KNOWN_RUNNERS = {
    Path("BalloonPoppingGymEnv/evaluation/evaluate.py"): "evaluate_scenario",
    Path("doc/examples/run_env_agent.py"): "run_for_development",
    Path("doc/examples/test_navigation_agent.py"): None,
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

# The mirror of the above for a loop written as ``while True`` with the flags
# checked in the body: leave as soon as either one is set.
LEAVE_NOW = {state: not keeps_going for state, keeps_going in KEEP_GOING.items()}

# What a runner calls the environment when nothing in the file binds it. Kept so
# a snippet, a docstring example or a helper that takes ``env`` as an argument is
# still read as a runner without having to construct anything first.
PLAIN_ENV_NAMES = frozenset({"env"})

ENV_CLASS_NAME = "BalloonPoppingEnv"
GYM_MODULE_NAMES = frozenset({"gym", "gymnasium"})

# Scopes of their own, so a name written inside one is not the loop's business.
# Only ``FunctionDef``/``AsyncFunctionDef``/``ClassDef`` can hold an assignment
# today; the lambda and the comprehensions are listed so that need not stay true.
NESTED_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

# ``ast.While`` too, or an outer ``while episodes_remaining:`` inherits the inner
# loop's step. ``for`` is deliberately not one: making it a barrier reports the
# two correct loops in tests/test_coordinate_contract.py, which leave by raising.
STEP_BARRIERS = NESTED_SCOPES + (ast.While,)


class EpisodeLoop(NamedTuple):
    """A ``while`` that steps an environment, with what it took to recognise it.
    ``env_names`` travels with the loop so ``unpack_problem`` finds the same step
    assignments discovery did, rather than none at all under some other name."""

    node: ast.While
    env_names: frozenset
    scope: str | None

    @property
    def lineno(self):
        return self.node.lineno


def _receiver_name(node):
    """The identifier a receiver expression hangs off, or None. ``self.env``
    answers ``env`` and ``envs[0]`` answers ``envs``, so the name that was bound
    is what gets compared."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _receiver_name(node.value)
    return None


def _is_env_constructor(node):
    """``gym.make(...)`` or ``BalloonPoppingEnv(...)``."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == ENV_CLASS_NAME
    if isinstance(node.func, ast.Attribute) and node.func.attr == "make":
        return _receiver_name(node.func.value) in GYM_MODULE_NAMES
    return False


def _bound_names(target, value):
    """The names ``target`` binds, if ``value`` is where an environment comes
    from. A constructor anywhere in ``value`` counts, so ``TimeLimit(gym.make())``
    does; a tuple target is paired off, so ``env, agent = ...`` binds only env."""
    if isinstance(target, (ast.Tuple, ast.List)):
        if not isinstance(value, (ast.Tuple, ast.List)):
            return []
        if len(target.elts) != len(value.elts):
            return []
        return [
            name
            for element, item in zip(target.elts, value.elts)
            for name in _bound_names(element, item)
        ]
    if not any(_is_env_constructor(inner) for inner in ast.walk(value)):
        return []
    name = _receiver_name(target)
    return [name] if name else []


def environment_names(tree):
    """Every name this module binds an environment to. Collected for the module
    rather than per scope, so an unrelated ``.step()`` on a name some other
    function used for the environment gets asked for the flags."""
    names = set(PLAIN_ENV_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_bound_names(target, node.value))
        elif (
            isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None
        ):
            names.update(_bound_names(node.target, node.value))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            names.update(_bound_names(node.optional_vars, node.context_expr))
    return frozenset(names)


def _is_env_step_call(node, env_names):
    """A call to ``.step()`` on something this module bound an environment to,
    so a future ``optimizer.step()`` or ``scheduler.step()`` inside a ``while``
    is not conscripted into being an episode loop."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
    ):
        return False
    return _receiver_name(node.func.value) in env_names


def _nodes_of_the_loop(loop):
    """Everything below ``loop`` that is the loop's own, stopping at the
    barriers. ``ast.walk`` crosses every scope boundary there is, so a step
    inside a nested ``def`` or ``while`` counted as the outer loop's own."""
    nodes = []

    def descend(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, STEP_BARRIERS):
                continue
            nodes.append(child)
            descend(child)

    descend(loop)
    return nodes


def _step_assignments(loop, env_names):
    """Assignments the loop itself makes whose value is a direct environment step."""
    return [
        node
        for node in _nodes_of_the_loop(loop)
        if isinstance(node, ast.Assign) and _is_env_step_call(node.value, env_names)
    ]


def episode_loops(tree):
    """Every ``while`` whose own body assigns from an environment step."""
    env_names = environment_names(tree)
    loops = []

    def descend(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.While) and _step_assignments(child, env_names):
                loops.append(EpisodeLoop(child, env_names, scope))
            inner_scope = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else scope
            )
            descend(child, inner_scope)

    descend(tree, None)
    return loops


def missing_runners(loops):
    """Known runners whose own stepping loop is not in ``loops``. Pinned to
    (path, enclosing function), so a decoy episode loop elsewhere in the same
    file cannot stand in for a runner that has stopped being discovered."""
    found = {(path, loop.scope) for path, loop in loops}
    return sorted(
        path for path, scope in KNOWN_RUNNERS.items() if (path, scope) not in found
    )


def _behaviour(expression):
    """How ``expression`` reads over the four flag combinations, or a complaint.
    The flags go in as globals: a ``lambda`` in the condition resolves names
    through globals at call time, and as locals it raised NameError."""
    names = {inner.id for inner in ast.walk(expression) if isinstance(inner, ast.Name)}
    if not names:
        return None, "never looks at either flag"
    if not names <= set(FLAG_NAMES):
        others = sorted(names - set(FLAG_NAMES))
        return None, f"also depends on {others}, which this cannot evaluate"
    if names != set(FLAG_NAMES):
        return None, f"only looks at {sorted(names)}"

    tree = ast.Expression(body=expression)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<loop condition>", "eval")
    behaviour = {}
    for terminated in (False, True):
        for truncated in (False, True):
            environment = {
                "__builtins__": {},
                "terminated": terminated,
                "truncated": truncated,
            }
            try:
                value = eval(code, environment)  # noqa: S307
            except Exception as error:  # noqa: BLE001
                # Reported rather than swallowed. Anything the two flags can
                # still raise, such as an attribute taken off one of them, has
                # to come back as a verdict instead of ending the run.
                return None, f"raised {type(error).__name__} when evaluated: {error}"
            behaviour[(terminated, truncated)] = bool(value)
    return behaviour, None


def _leaves_the_loop(body):
    """Whether this branch stops the loop outright."""
    return any(isinstance(statement, (ast.Break, ast.Return)) for statement in body)


def _exit_problem(loop):
    """What is wrong with how a ``while True`` leaves on the flags, or None. Two
    loops here step first and leave after, so a termination regression fails with
    a number. Only the ``if`` exits the body runs every time round count."""
    for statement in loop.node.body:
        if not isinstance(statement, ast.If) or not _leaves_the_loop(statement.body):
            continue
        behaviour, _complaint = _behaviour(statement.test)
        if behaviour == LEAVE_NOW:
            return None
    return (
        "the loop runs until its body leaves it, and nothing it runs every time "
        "round leaves on exactly (terminated or truncated)"
    )


def guard_problem(loop):
    """What is wrong with this loop's condition, or None. Judged by what it does
    rather than by what it says: ``not (terminated and truncated)`` mentions both
    names and keeps stepping after either one fires."""
    test = loop.node.test
    if isinstance(test, ast.Constant) and test.value:
        return _exit_problem(loop)

    behaviour, complaint = _behaviour(test)
    if complaint is not None:
        return f"the condition {complaint}"
    if behaviour != KEEP_GOING:
        keeps_going = sorted(state for state, goes in behaviour.items() if goes)
        return (
            f"the condition keeps stepping for (terminated, truncated) in "
            f"{keeps_going}; it has to keep going only for (False, False)"
        )
    return None


# The names the loop reads to decide whether to go round again. Writing to
# either between the step and the next evaluation makes the guard read something
# other than what the environment said.
DECISION_NAMES = frozenset({"terminated", "truncated"})


def _rechecks_and_leaves(statement):
    """Whether this ``if`` reads both decision names and leaves on them.

    A loop may step again in the same pass as long as it has looked at what the
    last step said first, which is what the condition would have done.
    """
    mentioned = {
        node.id
        for node in ast.walk(statement.test)
        if isinstance(node, ast.Name) and node.id in DECISION_NAMES
    }
    return mentioned == DECISION_NAMES and _leaves_the_loop(statement.body)


def _steps_along_the_longest_path(body, env_names):
    """The most environment steps one pass through ``body`` can make.

    Branches are the maximum of their arms rather than the sum, since only one
    arm runs; statements in sequence add up. Anything past a ``break``,
    ``return`` or ``raise`` is not reached and does not count.
    """
    total = 0
    for statement in body:
        if isinstance(statement, ast.Assign) and _is_env_step_call(
            statement.value, env_names
        ):
            total += 1
            continue
        if isinstance(statement, ast.Expr) and _is_env_step_call(
            statement.value, env_names
        ):
            total += 1
            continue
        if isinstance(statement, STEP_BARRIERS):
            # A nested function or loop is not this loop's own pass.
            continue
        if isinstance(statement, (ast.Break, ast.Return, ast.Raise)):
            break
        if isinstance(statement, ast.If):
            if _rechecks_and_leaves(statement):
                # The flags have been read again and the loop left if either
                # fired, so anything after this is a fresh pass, not a second
                # step on a stale answer.
                total = 0
                continue
            total += max(
                _steps_along_the_longest_path(statement.body, env_names),
                _steps_along_the_longest_path(statement.orelse, env_names),
            )
            continue
        if isinstance(statement, (ast.Try, ast.With)):
            total += _steps_along_the_longest_path(statement.body, env_names)
            for handler in getattr(statement, "handlers", ()):
                total += _steps_along_the_longest_path(handler.body, env_names)
            total += _steps_along_the_longest_path(
                getattr(statement, "orelse", []), env_names
            )
            total += _steps_along_the_longest_path(
                getattr(statement, "finalbody", []), env_names
            )
            continue
    return total


def second_step_problem(loop):
    """Whether one pass can step more than once, or None.

    The condition is only read between passes. A second step in the same pass
    runs whatever the first one said, which is the thing this file is about.
    """
    most = _steps_along_the_longest_path(loop.node.body, loop.env_names)
    if most > 1:
        return (
            f"one pass through the loop can call step() {most} times, and the "
            f"condition is only read between passes"
        )
    return None


def _writes_to_decision_names(statement):
    """The decision names this statement assigns to.

    Its own recursion rather than ``ast.walk``, which yields every descendant
    before anything can decline one, so a guard written as a ``continue`` there
    prunes nothing. A ``lambda`` is stepped over because a walrus inside one
    binds in the lambda; a comprehension is not, because a walrus inside one
    binds out here, which is the whole reason it is worth looking for.
    """
    written = set()

    def visit(node):
        if isinstance(node, ast.Lambda):
            return
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and name.id in DECISION_NAMES:
                    written.add(name.id)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(statement)
    return written


def overwritten_flag_problem(loop):
    """Whether the loop writes a decision name other than from a step, or None.

    ``terminated = truncated = False`` after the unpack leaves the guard reading
    what the loop decided rather than what the environment reported, and the
    loop never ends.
    """
    from_step = {id(node) for node in _step_assignments(loop.node, loop.env_names)}
    for statement in _nodes_of_the_loop(loop.node):
        if not isinstance(
            statement, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr)
        ):
            continue
        if id(statement) in from_step:
            continue
        written = _writes_to_decision_names(statement)
        if written:
            return (
                f"the loop assigns {', '.join(sorted(written))} from something "
                f"other than step(), so the condition stops reading what the "
                f"environment reported"
            )
    return None


def unpack_problem(loop):
    """What is wrong with how this loop takes ``step()`` apart, or None."""
    for assignment in _step_assignments(loop.node, loop.env_names):
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


# CommonMark spells a fence as at least three backticks or at least three
# tildes, and the closer has to be the same character and no shorter.
FENCE_CHARACTERS = ("`", "~")
MINIMUM_FENCE_LENGTH = 3

# The language is compared lowercased, and nothing else is added: ``py3`` and
# ``{python}`` are still not Python blocks as far as this is concerned.
PYTHON_FENCE_LANGUAGES = frozenset({"python", "py"})


def _fence_run(stripped):
    """The opening or closing fence this line is, or None."""
    if not stripped or stripped[0] not in FENCE_CHARACTERS:
        return None
    length = len(stripped) - len(stripped.lstrip(stripped[0]))
    return stripped[:length] if length >= MINIMUM_FENCE_LENGTH else None


def _fence_language(stripped, run):
    """The language word of an info string, lowercased. Everything after the
    first word is the rest of the info string, which CommonMark allows and this
    only has to get out of the way of."""
    words = stripped[len(run) :].strip().split()
    return words[0].lower() if words else ""


def source_files():
    """Every Python file in this repository. ``os.walk`` rather than ``rglob``
    so the excluded directories are pruned before they are opened, and the
    dotted ones (``.git``, ``.venv``, the caches) with them."""
    for directory, subdirectories, filenames in os.walk(REPO_ROOT):
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if not name.startswith(".")
            and not name.endswith(".egg-info")
            and name not in EXCLUDED_DIRECTORIES
        )
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield Path(directory, filename).relative_to(REPO_ROOT)


class TestTheRunnersInThisRepository(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loops = []
        cls.scanned = list(source_files())
        for path in cls.scanned:
            # Deliberately not catching SyntaxError. A file that stops parsing
            # would otherwise drop out of discovery while everything stayed
            # green, and not every directory scanned here is one CI lints.
            source = (REPO_ROOT / path).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for loop in episode_loops(tree):
                cls.loops.append((path, loop))

    def test_the_known_runners_were_all_found(self):
        """Or every assertion below holds over whatever is left.

        The discovery is the part that can quietly stop working: a rename, a
        move, or narrowing what counts as a step call.
        """
        missing = missing_runners(self.loops)

        self.assertEqual(missing, [], f"these runners were not discovered: {missing}")

    def test_the_scan_reaches_the_tests_directory(self):
        """It used to stop at three directories, and this one was not among them."""
        self.assertTrue(
            any(path.parts[0] == "tests" for path, _ in self.loops),
            "no loop under tests/ was scanned",
        )

    def test_the_scan_skips_the_directories_that_are_not_ours(self):
        skipped = {".venv", "ActiveRocketPy", ".git"}

        self.assertEqual(
            [path for path in self.scanned if set(path.parts) & skipped], []
        )

    def test_every_loop_stops_when_either_flag_is_set(self):
        for path, loop in self.loops:
            with self.subTest(runner=f"{path}:{loop.lineno}"):
                self.assertIsNone(guard_problem(loop), f"{path} line {loop.lineno}")

    def test_every_loop_takes_both_flags_out_of_the_step_result(self):
        for path, loop in self.loops:
            with self.subTest(runner=f"{path}:{loop.lineno}"):
                self.assertIsNone(unpack_problem(loop), f"{path} line {loop.lineno}")

    def test_no_loop_steps_twice_on_one_answer(self):
        for path, loop in self.loops:
            with self.subTest(runner=f"{path}:{loop.lineno}"):
                self.assertIsNone(
                    second_step_problem(loop), f"{path} line {loop.lineno}"
                )

    def test_no_loop_writes_over_what_the_environment_said(self):
        for path, loop in self.loops:
            with self.subTest(runner=f"{path}:{loop.lineno}"):
                self.assertIsNone(
                    overwritten_flag_problem(loop), f"{path} line {loop.lineno}"
                )


class TestASecondStepOnTheSameAnswer(unittest.TestCase):
    """The condition is read between passes, not during one.

    Both shapes here passed every check this file had: they take the flags apart
    correctly and their condition is exactly right, and they step again anyway.
    """

    HEAD = (
        "def run(env, action, debug):\n"
        "    terminated = truncated = False\n"
        "    while not (terminated or truncated):\n"
    )
    STEP = "        o, r, terminated, truncated, i = env.step(action)\n"

    def _only_loop(self, body):
        loops = episode_loops(ast.parse(self.HEAD + body))
        self.assertEqual(len(loops), 1)
        return loops[0]

    def test_one_step_is_fine(self):
        self.assertIsNone(second_step_problem(self._only_loop(self.STEP)))

    def test_two_steps_in_a_row_are_not(self):
        problem = second_step_problem(self._only_loop(self.STEP + self.STEP))

        self.assertIn("2 times", problem)

    def test_a_second_step_under_a_condition_is_not(self):
        body = self.STEP + "        if debug:\n    " + self.STEP

        self.assertIsNotNone(second_step_problem(self._only_loop(body)))

    def test_one_step_in_each_arm_is_fine(self):
        """Only one arm runs, so a pass still makes one step."""
        body = "        if debug:\n    " + self.STEP + "        else:\n    " + self.STEP

        self.assertIsNone(second_step_problem(self._only_loop(body)))

    def test_stepping_again_after_reading_the_flags_is_fine(self):
        """The check is about a stale answer, not about the number of calls."""
        body = (
            self.STEP
            + "        if terminated or truncated:\n            break\n"
            + self.STEP
        )

        self.assertIsNone(second_step_problem(self._only_loop(body)))

    def test_reading_only_one_flag_before_stepping_again_is_not(self):
        body = self.STEP + "        if terminated:\n            break\n" + self.STEP

        self.assertIsNotNone(second_step_problem(self._only_loop(body)))

    def test_reading_the_flags_without_leaving_is_not(self):
        body = (
            self.STEP
            + "        if terminated or truncated:\n            pass\n"
            + self.STEP
        )

        self.assertIsNotNone(second_step_problem(self._only_loop(body)))

    def test_a_step_in_a_nested_loop_is_not_this_loops_pass(self):
        """The inner ``while`` is its own episode loop and is judged separately.

        Counting its step as the outer loop's would report a second step on the
        same answer where there is none.
        """
        body = self.STEP + "        while debug:\n    " + self.STEP
        loops = episode_loops(ast.parse(self.HEAD + body))
        self.assertEqual(len(loops), 2)
        outer = min(loops, key=lambda loop: loop.lineno)

        self.assertIsNone(second_step_problem(outer))


class TestWritingOverWhatTheEnvironmentSaid(unittest.TestCase):
    """A guard that reads what the loop decided is not a guard.

    ``terminated = truncated = False`` after the unpack passed every check this
    file had, and never ends.
    """

    HEAD = TestASecondStepOnTheSameAnswer.HEAD
    STEP = TestASecondStepOnTheSameAnswer.STEP

    def _only_loop(self, body):
        loops = episode_loops(ast.parse(self.HEAD + body))
        self.assertEqual(len(loops), 1)
        return loops[0]

    def test_the_step_unpack_itself_is_not_an_overwrite(self):
        self.assertIsNone(overwritten_flag_problem(self._only_loop(self.STEP)))

    def test_clearing_both_flags_is(self):
        body = self.STEP + "        terminated = truncated = False\n"

        problem = overwritten_flag_problem(self._only_loop(body))

        self.assertIn("terminated", problem)
        self.assertIn("truncated", problem)

    def test_narrowing_one_flag_is(self):
        body = self.STEP + "        terminated = bool(terminated and debug)\n"

        self.assertIsNotNone(overwritten_flag_problem(self._only_loop(body)))

    def test_an_augmented_assignment_is(self):
        body = self.STEP + "        truncated |= debug\n"

        self.assertIsNotNone(overwritten_flag_problem(self._only_loop(body)))

    def test_a_walrus_is(self):
        body = self.STEP + "        print(truncated := debug)\n"

        self.assertIsNotNone(overwritten_flag_problem(self._only_loop(body)))

    def test_writing_an_unrelated_name_is_not(self):
        """The control: this check has to leave ordinary loop bodies alone."""
        body = self.STEP + "        steps = steps + 1\n"

        self.assertIsNone(overwritten_flag_problem(self._only_loop(body)))

    def test_a_write_inside_a_nested_function_is_not_this_loops(self):
        body = self.STEP + "        def reset_them():\n            terminated = False\n"

        self.assertIsNone(overwritten_flag_problem(self._only_loop(body)))

    def test_a_walrus_inside_a_lambda_is_not_this_loops(self):
        """It binds in the lambda. The first version of this reported it, since
        `ast.walk` hands back every descendant before anything can decline one,
        so declining with `continue` there pruned nothing."""
        body = self.STEP + "        f = lambda: (terminated := False)\n"

        self.assertIsNone(overwritten_flag_problem(self._only_loop(body)))

    def test_a_walrus_inside_a_comprehension_is_this_loops(self):
        """That one binds out here, which is why the lambda case cannot be
        handled by skipping nested scopes as a group."""
        body = self.STEP + "        x = [(truncated := False) for _ in range(1)]\n"

        self.assertIsNotNone(overwritten_flag_problem(self._only_loop(body)))


class TestTheLoopTheReadmeShows(unittest.TestCase):
    """The README snippet is what a competitor copies, and it is not a .py file.

    So the scan above cannot see it, and it went stale in exactly the way the
    runners did: it described a single "termination flag" and said the episode
    ends when the time limit is reached or the rocket lands, without saying that
    those are now two different flags and that a loop has to watch both.
    """

    @staticmethod
    def _python_blocks(text):
        """The Python fenced blocks in ``text``. An unclosed fence used to
        return nothing, so the README test ran over zero blocks and passed;
        ``Python``, ``python title=x`` and ``~~~python`` were read as prose."""
        blocks = []
        collecting = None
        fence = None
        opened_at = 0
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            run = _fence_run(stripped)
            if collecting is None:
                if run and _fence_language(stripped, run) in PYTHON_FENCE_LANGUAGES:
                    collecting = []
                    fence = run
                    opened_at = number
                continue
            closes = (
                run
                and stripped == run
                and run[0] == fence[0]
                and len(run) >= len(fence)
            )
            if closes:
                blocks.append("\n".join(collecting))
                collecting = None
            else:
                # Snippets in the README are indented under a list item.
                collecting.append(line[3:] if line.startswith("   ") else line)
        if collecting is not None:
            raise ValueError(
                f"the {fence} block opened at line {opened_at} is never closed, "
                "so it would go unchecked"
            )
        return blocks

    def test_an_unclosed_fence_fails_loudly(self):
        with self.assertRaises(ValueError):
            self._python_blocks("```python\nwhile not terminated:\n    pass\n")

    def test_the_language_tag_is_matched_without_regard_to_case(self):
        blocks = self._python_blocks("```Python\nx = 1\n```\n")

        self.assertEqual(blocks, ["x = 1"])

    def test_an_info_string_after_the_language_still_opens_a_block(self):
        blocks = self._python_blocks("```python title=loop.py\nx = 1\n```\n")

        self.assertEqual(blocks, ["x = 1"])

    def test_a_tilde_fence_opens_a_block(self):
        blocks = self._python_blocks("~~~python\nx = 1\n~~~\n")

        self.assertEqual(blocks, ["x = 1"])

    def test_another_language_is_still_not_collected(self):
        for text in ("```shell\nx = 1\n```\n", "```py3\nx = 1\n```\n"):
            with self.subTest(text=text):
                self.assertEqual(self._python_blocks(text), [])

    def test_the_readme_shows_an_episode_loop(self):
        """Or the check below holds over nothing at all."""
        blocks = self._python_blocks((REPO_ROOT / "README.md").read_text("utf-8"))
        loops = [loop for block in blocks for loop in episode_loops(ast.parse(block))]

        self.assertTrue(loops, "the README shows no loop driving env.step()")

    def test_the_loop_the_readme_shows_is_one_that_works(self):
        blocks = self._python_blocks((REPO_ROOT / "README.md").read_text("utf-8"))
        for block in blocks:
            for loop in episode_loops(ast.parse(block)):
                with self.subTest(line=loop.lineno):
                    self.assertIsNone(guard_problem(loop))
                    self.assertIsNone(unpack_problem(loop))


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


class TestTheCheckerFindsTheEnvironmentItWasBoundTo(unittest.TestCase):
    """Which name the environment is under is the author's choice, not ours."""

    RENAMED = (
        "world = gym.make('BalloonPoppingGymEnv/BalloonPoppingEnv-v0')\n"
        "while not (terminated or truncated):\n"
        "    o, r, terminated, truncated, i = world.step(a)",
        "e = gym.make('BalloonPoppingGymEnv/BalloonPoppingEnv-v0')\n"
        "while not (terminated or truncated):\n"
        "    o, r, terminated, truncated, i = e.step(a)",
        "self.world = BalloonPoppingEnv(render_mode=None, parameters=p)\n"
        "while not (terminated or truncated):\n"
        "    o, r, terminated, truncated, i = self.world.step(a)",
        "envs = [BalloonPoppingEnv(render_mode=None, parameters=p)]\n"
        "while not (terminated or truncated):\n"
        "    o, r, terminated, truncated, i = envs[0].step(a)",
    )

    def test_a_renamed_environment_is_still_found(self):
        for source in self.RENAMED:
            with self.subTest(source=source.splitlines()[0]):
                loops = episode_loops(ast.parse(source))

                self.assertEqual(len(loops), 1)
                self.assertIsNone(guard_problem(loops[0]))
                self.assertIsNone(unpack_problem(loops[0]))

    def test_the_plain_names_keep_working_without_any_binding(self):
        for receiver in ("env", "self.env"):
            source = (
                "while not (terminated or truncated):\n"
                f"    o, r, terminated, truncated, i = {receiver}.step(a)"
            )
            with self.subTest(receiver=receiver):
                self.assertEqual(len(episode_loops(ast.parse(source))), 1)

    def test_a_name_that_merely_contains_env_is_not_the_environment(self):
        source = "while sampling:\n    level = envelope_follower.step(sample)"

        self.assertEqual(episode_loops(ast.parse(source)), [])


class TestTheCheckerStaysInsideTheLoopItIsJudging(unittest.TestCase):
    """A ``while`` answers for what its own body does, not for what is under it."""

    MULTI_EPISODE = (
        "while episodes_remaining:\n"
        "    terminated = False\n"
        "    truncated = False\n"
        "    while not (terminated or truncated):\n"
        "        o, r, terminated, truncated, i = env.step(a)\n"
        "    episodes_remaining -= 1"
    )

    def test_only_the_inner_loop_of_a_multi_episode_runner_is_an_episode_loop(self):
        loops = episode_loops(ast.parse(self.MULTI_EPISODE))

        self.assertEqual([loop.lineno for loop in loops], [4])
        self.assertIsNone(guard_problem(loops[0]))
        self.assertIsNone(unpack_problem(loops[0]))

    def test_a_step_in_a_nested_scope_does_not_belong_to_the_loop(self):
        for source in (
            "while running:\n"
            "    def one_episode():\n"
            "        o, r, terminated, truncated, i = env.step(a)",
            "while running:\n"
            "    async def one_episode():\n"
            "        o, r, terminated, truncated, i = env.step(a)",
            "while running:\n    class Runner:\n        result = env.step(a)",
        ):
            with self.subTest(source=source.splitlines()[1].strip()):
                self.assertEqual(episode_loops(ast.parse(source)), [])

    def test_a_guard_that_wraps_the_flags_in_a_lambda_is_evaluated(self):
        source = (
            "while (lambda: not (terminated or truncated))():\n"
            "    o, r, terminated, truncated, i = env.step(a)"
        )
        loop = episode_loops(ast.parse(source))[0]

        self.assertIsNone(guard_problem(loop))

    def test_a_guard_that_raises_when_evaluated_is_reported_not_crashed(self):
        """A verdict, not a traceback. One bad guard used to end the whole run."""
        source = (
            "while not terminated.pending or truncated:\n"
            "    o, r, terminated, truncated, i = env.step(a)"
        )
        loop = episode_loops(ast.parse(source))[0]

        self.assertIsNotNone(guard_problem(loop))

    def test_a_wrong_guard_wrapped_in_a_lambda_is_still_rejected(self):
        source = (
            "while (lambda: not (terminated and truncated))():\n"
            "    o, r, terminated, truncated, i = env.step(a)"
        )
        loop = episode_loops(ast.parse(source))[0]

        self.assertIsNotNone(guard_problem(loop))


class TestALoopThatLeavesFromItsBody(unittest.TestCase):
    """``while True`` with the flags checked after the step, which this repo uses."""

    STEP = "    o, r, terminated, truncated, i = env.step(a)\n"

    def _only_loop(self, body):
        loops = episode_loops(ast.parse("while True:\n" + self.STEP + body))
        assert len(loops) == 1, f"expected one episode loop, found {len(loops)}"
        return loops[0]

    def test_leaving_on_either_flag_is_accepted(self):
        for leave in ("break", "return steps"):
            with self.subTest(leave=leave):
                loop = self._only_loop(
                    f"    if terminated or truncated:\n        {leave}"
                )

                self.assertIsNone(guard_problem(loop))

    def test_leaving_on_one_flag_only_is_rejected(self):
        loop = self._only_loop("    if terminated:\n        break")

        self.assertIsNotNone(guard_problem(loop))

    def test_never_leaving_on_the_flags_is_rejected(self):
        loop = self._only_loop("    steps += 1")

        self.assertIsNotNone(guard_problem(loop))

    def test_a_break_inside_a_nested_loop_does_not_end_the_outer_one(self):
        loop = self._only_loop(
            "    for _ in range(2):\n"
            "        if terminated or truncated:\n"
            "            break"
        )

        self.assertIsNotNone(guard_problem(loop))


class TestTheDiscoveryPin(unittest.TestCase):
    """The pin has to name the loop, not just the file it lives in."""

    @staticmethod
    def _loop_in(scope):
        """An episode loop written at module level, or inside ``scope``."""
        source = (
            "while not (terminated or truncated):\n"
            "    o, r, terminated, truncated, i = env.step(a)"
        )
        if scope is not None:
            source = f"def {scope}():\n    " + source.replace("\n", "\n    ")
        return episode_loops(ast.parse(source))[0]

    def _everything_found(self):
        return [(path, self._loop_in(scope)) for path, scope in KNOWN_RUNNERS.items()]

    def test_the_pin_is_satisfied_by_the_loop_it_names(self):
        self.assertEqual(missing_runners(self._everything_found()), [])

    def test_a_decoy_loop_in_the_same_file_does_not_stand_in_for_the_runner(self):
        found = self._everything_found()
        path, _real = found[0]
        found[0] = (path, self._loop_in("somewhere_else"))

        self.assertEqual(missing_runners(found), [path])


class TestHowALoopIsAllowedToLeave(unittest.TestCase):
    """Two branches of the ``while True`` exit rule that nothing failed on. Each
    admits a genuinely broken loop when it breaks, and both survived a full run
    of this file before they were pinned here."""

    def problem(self, body):
        source = (
            "import gymnasium as gym\nenv = gym.make('x')\nwhile True:\n"
            "    o, r, terminated, truncated, i = env.step(a)\n" + body
        )
        (loop,) = episode_loops(ast.parse(source))
        return guard_problem(loop)

    def test_setting_a_variable_is_not_leaving_the_loop(self):
        """``running = False`` does not end a ``while True``. With the
        break-or-return test widened to accept any statement, this passed."""
        self.assertIsNotNone(
            self.problem("    if terminated or truncated:\n        running = False\n")
        )

    def test_leaving_only_when_both_flags_are_set_is_leaving_never(self):
        """The exit has to mirror the guard, not mention both names. ``and``
        here keeps stepping after either one fires, and comparing the behaviour
        to anything other than ``LEAVE_NOW`` lets it through."""
        self.assertIsNotNone(
            self.problem("    if terminated and truncated:\n        break\n")
        )

    def test_the_ordinary_form_is_still_accepted(self):
        """The half that stops the two above being satisfied by refusing
        everything."""
        self.assertIsNone(
            self.problem("    if terminated or truncated:\n        break\n")
        )


if __name__ == "__main__":
    unittest.main()
