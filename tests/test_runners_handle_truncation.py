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

What a checker gets wrong is what it cannot see
-----------------------------------------------
The three things below were each measured, and each one is a loop going
unchecked rather than a loop reported wrongly, which is the failure a checker
can carry indefinitely without anyone noticing.

* Which name the environment is under is the author's choice. The receiver used
  to be matched with ``"env" in name.lower()``, so ``world = gym.make(...)`` was
  never discovered and never checked, while ``envelope_follower.step(sample)``
  was reported for flags it has nothing to do with. Resolved against what the
  module actually binds now.
* A ``while`` answers for its own body. The step assignments were collected with
  ``ast.walk``, which crosses into a nested ``def`` and into a nested ``while``,
  so ``while episodes_remaining:`` around ``while not (terminated or
  truncated):`` was reported for the inner loop's step.
* What the scan does not open cannot fail it. It covered three directories, and
  ``tests/`` was not one of them; two loops there were waiting on ``terminated``
  alone when it was widened to the repository.

AST only, so it runs without the simulation stack.
"""

import ast
import os
import unittest
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything under the repository root except these. The previous list was
# ``("BalloonPoppingGymEnv", "doc", "scripts")``, which left ``tests/`` and the
# root itself unscanned, and two loops in ``tests/`` were waiting on
# ``terminated`` alone at the time this was widened. Pruned during the walk
# rather than filtered afterwards, so ``.venv`` and the ActiveRocketPy submodule
# are never opened: the whole scan is 46 files in 0.03 s that way. ActiveRocketPy
# is a fork of RocketPy with its own runners, and this file has no say over them.
EXCLUDED_DIRECTORIES = frozenset(
    {".venv", ".ci-venv", "ActiveRocketPy", "build", "dist", "node_modules"}
)

# The loops that exist today, each pinned to the function it lives in (``None``
# for one written at module level). The pin used to be the path alone, which
# only asked whether *some* episode loop had been found in that file: a decoy
# loop anywhere else in it satisfied that while the real runner had stopped
# being discovered and was going unchecked. Adding a runner is still fine,
# losing one silently is not.
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
# today; a lambda and the comprehensions are listed so the barrier does not
# depend on that staying true.
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

# A step under a nested ``while`` belongs to that ``while``, which is judged on
# its own. Without this an outer ``while episodes_remaining:`` inherits the inner
# loop's step and gets asked for flags that are the inner loop's job, which is
# how the idiomatic multi-episode runner came out reported.
#
# ``for`` is not a barrier, and the reason is narrower than "a ``for`` is never
# an episode loop". Sometimes it is. A bounded runner,
#
#     while retries:
#         for _ in range(max_steps):
#             o, r, terminated, truncated, i = env.step(a)
#             if terminated or truncated:
#                 break
#
# is correct and gets its enclosing ``while retries:`` reported for a condition
# that belongs to the inner loop. That is a known false positive, and a bounded
# runner written without any enclosing ``while`` is not discovered at all.
#
# The repair, making ``for`` a barrier and judging a stepping ``for`` on its
# exits the way ``while True`` is judged, was tried and is worse here. It reports
# the two loops in tests/test_coordinate_contract.py, which step inside
# ``for _ in range(600)`` and answer both flags with ``self.fail(...)``. Leaving
# by raising is a real exit that ``_leaves_the_loop`` does not model, and
# widening it to accept any call would accept most things. Neither missing shape
# exists in this repository, and two correct loops do, so it stays as it is until
# the exit rule can read a raise.
STEP_BARRIERS = NESTED_SCOPES + (ast.While,)


class EpisodeLoop(NamedTuple):
    """A ``while`` that steps an environment, with what it took to recognise it.

    ``env_names`` travels with the loop because ``unpack_problem`` has to find
    the same step assignments the discovery did. Handing it a loop without them
    made it look at a runner that binds the environment to some other name, find
    no step call, and report nothing wrong.
    """

    node: ast.While
    env_names: frozenset
    scope: str | None

    @property
    def lineno(self):
        return self.node.lineno


def _receiver_name(node):
    """The identifier a receiver expression hangs off, or None.

    ``self.env`` answers ``env`` and ``envs[0]`` answers ``envs``, so the name
    that was bound is what gets compared.
    """
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
    """The names ``target`` binds, if ``value`` is where an environment comes from.

    The test is whether a constructor appears anywhere in ``value`` rather than
    whether ``value`` *is* one, which is what lets a list of them and a wrapped
    ``env = TimeLimit(gym.make(...))`` be recognised by the same rule. A tuple
    target is paired off element by element, so ``env, agent = gym.make(...),
    Agent()`` does not hand the agent the environment's name.
    """
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
    """Every name this module binds an environment to.

    Matching the receiver by substring was the whole check before, and it read
    both ways: ``world = gym.make(...)`` was never discovered at all, so it was
    never checked, and an unrelated ``envelope_follower.step(sample)`` was
    reported for flags it has no reason to have.

    Collected for the module rather than per scope, so a name one function binds
    counts in another. The alternative is a scope analysis, and what being
    generous costs is that an unrelated ``.step()`` on a name some other function
    used for the environment gets asked for the flags.
    """
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
    """A call to ``.step()`` on something this module bound an environment to.

    The receiver is resolved so that a future ``optimizer.step()`` or
    ``scheduler.step()`` inside a ``while`` is not conscripted into being an
    episode loop and asked for flags it has no reason to have.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
    ):
        return False
    return _receiver_name(node.func.value) in env_names


def _nodes_of_the_loop(loop):
    """Everything below ``loop`` that is the loop's own, stopping at the barriers.

    ``ast.walk`` was used here and crosses every scope boundary there is, so a
    step written inside a nested ``def`` or under a nested ``while`` counted as
    the outer loop's own.
    """
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
    """Known runners whose own stepping loop is not in ``loops``.

    Pinned to (path, enclosing function) rather than to the path, so a decoy
    episode loop elsewhere in the same file cannot stand in for a runner that
    has stopped being discovered.
    """
    found = {(path, loop.scope) for path, loop in loops}
    return sorted(
        path for path, scope in KNOWN_RUNNERS.items() if (path, scope) not in found
    )


def _behaviour(expression):
    """How ``expression`` reads over the four flag combinations, or a complaint.

    The flags go in as globals. They were locals, and a name inside a ``lambda``
    or a comprehension in the condition resolves through globals and builtins at
    call time, never through the locals ``eval`` was handed, so a guard written
    as ``(lambda: not (terminated or truncated))()`` raised NameError and took
    the run down instead of returning a verdict.
    """
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
    """What is wrong with how a ``while True`` leaves on the flags, or None.

    Two loops in this repository are written this way, for a reason they state:
    they step first and leave on the flags afterwards, so the step count can be
    checked against a bound on the way round and a termination regression fails
    with a number instead of hanging until the CI timeout. Reading only
    ``loop.test`` calls that "never looks at either flag", which is true of the
    text and wrong about the loop.

    Only the ``if`` statements the loop body runs every time round count. An exit
    nested inside another branch is reached conditionally, and one inside a
    nested ``for`` ends that ``for`` rather than this loop, so neither is an
    answer to whether *this* loop stops. A loop that puts its exit somewhere else
    is reported rather than guessed at.
    """
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
    """What is wrong with this loop's condition, or None.

    Judged by what it does rather than by what it says. ``not (terminated and
    truncated)`` mentions both names and keeps stepping after either one fires.
    """
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
    """The language word of an info string, lowercased.

    Everything after the first word is the rest of the info string, which
    CommonMark allows and this only has to get out of the way of.
    """
    words = stripped[len(run) :].strip().split()
    return words[0].lower() if words else ""


def source_files():
    """Every Python file in this repository.

    ``os.walk`` rather than ``rglob`` so the excluded directories are pruned
    before they are opened. Dotted directories go too: ``.git``, ``.venv`` and
    the caches are all of them, and none holds this repository's source.
    """
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


class TestTheLoopTheReadmeShows(unittest.TestCase):
    """The README snippet is what a competitor copies, and it is not a .py file.

    So the scan above cannot see it, and it went stale in exactly the way the
    runners did: it described a single "termination flag" and said the episode
    ends when the time limit is reached or the rocket lands, without saying that
    those are now two different flags and that a loop has to watch both.
    """

    @staticmethod
    def _python_blocks(text):
        """The Python fenced blocks in ``text``.

        Every part of this beyond the plain `````python`` case is here because
        the previous version failed silently rather than loudly. A fence left
        open ran off the end of the file with the block still in hand and
        returned it to nobody, so the README test below ran over zero blocks and
        passed for it; `````Python``, `````python title=x`` and ``~~~python``
        were each read as prose for the same reason. None of them says anything
        about the snippet being wrong, and all of them stop it being checked.
        """
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


if __name__ == "__main__":
    unittest.main()


class TestHowALoopIsAllowedToLeave(unittest.TestCase):
    """Two branches of the ``while True`` exit rule that nothing failed on.

    Each admits a genuinely broken loop when it breaks, so each is pinned here
    rather than left to the reader. Found by mutating the rule, not by reading
    it: both survived a full run of this file.
    """

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
        """The exit has to be the mirror of the guard rather than a mention of
        both names. ``and`` here keeps stepping after either one fires, which is
        the failure this file is about, and comparing the behaviour to anything
        other than ``LEAVE_NOW`` lets it through."""
        self.assertIsNotNone(
            self.problem("    if terminated and truncated:\n        break\n")
        )

    def test_the_ordinary_form_is_still_accepted(self):
        """The half that stops the two above being satisfied by refusing
        everything."""
        self.assertIsNone(
            self.problem("    if terminated or truncated:\n        break\n")
        )
