# Contributing

Thanks for taking the time. This describes how the repository actually works, so
you can get a change through review without guessing.

## If you are competing

The simulator is fixed for evaluation. Write your agent under
`BalloonPoppingGymEnv/agents/` and leave the rest alone; `pack_for_submission`
checks `evaluate.py` against the official copy and says so when they differ.
Questions about the rules or the environment belong in an issue rather than a
pull request.

## Setup

The physics engine lives in a submodule and a fresh clone leaves it empty, so
this comes first. Nothing that touches the simulator runs until it is populated,
which is everything except the AST-only tests:

```shell
git submodule update --init --recursive --checkout
```

`--checkout` is the default only while nothing sets
`submodule.ActiveRocketPy.update` locally; with that set to `merge` or `rebase`
the submodule stays wherever it already was. Same command as the README.

Then either path works:

```shell
uv sync --extra dev              # recommended; installs the pinned Python too
```

```shell
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

For the optional vpython renderer, ask for both extras in one command:
`uv sync --extra dev --extra vpython`. `uv sync` is exact, so running
`--extra vpython` on its own afterwards uninstalls everything that only the
`dev` extra brings in, ruff included. vpython is imported lazily on purpose, so
never import it at module scope.

## Branches

`develop` is where work lands. Branch from it, and open your pull request against
it. `main` is the released line, and cutting a release from `develop` is the
maintainer's call.

## Before you open a pull request

Run what CI runs:

```shell
uvx ruff@0.15.20 check BalloonPoppingGymEnv/ tests/
uvx ruff@0.15.20 format --check BalloonPoppingGymEnv/ tests/
uv sync --locked --extra dev
BPC_RUN_SLOW_TESTS=1 uv run --no-sync pytest tests/ --cov=BalloonPoppingGymEnv
```

`make format` fixes import order and formatting in place, using the same pinned
ruff.

There is one ruff version for the whole project: the CI pin, the `dev` extra, the
lockfile and `make format` all say 0.15.20, and `required-version` in
`pyproject.toml` makes any other release refuse to run rather than quietly apply
different rules:

```
ruff failed
  Cause: Required version `==0.15.20` does not match the running version `0.15.13`
```

So an editor integration or a stale environment tells you, instead of formatting a
file that CI then rejects.

`--locked` matters more than it looks. It fails when `uv.lock` has gone
stale against the ActiveRocketPy submodule, and it is what makes a green run a
statement about a known set of versions: CI installs from the lockfile too, so
what you run here and what CI runs are the same software.

What it does not check is the simulator's source. ActiveRocketPy is an editable
path dependency, so the lockfile records its version and dependencies with no
commit, and a source-only change leaves this green. The recorded submodule
commit is what pins that.

## Tests

The suite runs in two tiers, and both are worth keeping that way.

Some tests read the source with `ast` and never import it, so they run anywhere.
The rest need the simulation stack and skip cleanly when it is missing. Guard
only `import rocketpy`, not the package's own imports: a renamed symbol inside
this package should fail loudly rather than turn into a skip.

`BPC_RUN_SLOW_TESTS=1` enables the scenario-1 Monte Carlo golden master, which is
skipped by default so local runs stay fast. CI sets it.

Scenarios 0 and 1 are pinned against stored baselines in `tests/baselines/` so
that a change in ActiveRocketPy cannot move a competitor's score unnoticed. When
you change the physics on purpose, regenerate them deliberately and review the
diff:

```shell
PYTHONPATH=. python tests/baselines/regenerate_scenario_0.py
PYTHONPATH=. python tests/baselines/regenerate_scenario_1.py
```

Reading that diff takes a moment of care. Regenerating on a different machine
moves the last few significant digits even when nothing changed, because the
numbers come out of a different BLAS build:

```
-      20.002549511995664
+      20.002549509865123
```

That is the jitter the tolerance exists to absorb, and it is not a physics
change. A real one typically shows up in the leading digits, or in
`popped_count`, or in the step count. Typically, not always: a small real change
can sit in the low digits or stay inside the tolerance, so passing is not proof
that nothing moved. Those larger signals are a reason to look, not the only
place worth looking.

A baseline is tied to the pair of (repository, submodule commit). Regenerating
one because it went red, without understanding why, defeats the point of having
it.

## Reporting issues

Use the templates. A bug report that names the scenario, the seed and the
ActiveRocketPy commit is one somebody can reproduce; without them it usually
turns into a round trip.

## Changes worth a changelog entry

Anything a user or a contributor would notice goes in `CHANGELOG.md` under
`Unreleased`: environment behaviour, the action or observation space, the
submission format, scoring, the supported Python versions, or the tooling
needed to work on the repository. The release PR renames that section to the
version being cut and adds its comparison link, so nothing here carries a
placeholder date.
