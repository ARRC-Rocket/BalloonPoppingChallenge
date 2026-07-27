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
this comes first. Nothing imports until it is populated:

```shell
git submodule update --init --recursive
```

Then either path works:

```shell
uv sync --extra dev              # recommended; installs the pinned Python too
```

```shell
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`uv sync --extra vpython` adds the optional vpython renderer. It is imported
lazily on purpose, so never import it at module scope.

## Branches

`develop` is where work lands. Branch from it, and open your pull request against
it. `main` is the released line, and cutting a release from `develop` is the
maintainer's call.

## Before you open a pull request

Run what CI runs. The ruff version is pinned, and an unpinned one will disagree
about formatting:

```shell
uvx ruff@0.15.20 check BalloonPoppingGymEnv/ tests/
uvx ruff@0.15.20 format --check BalloonPoppingGymEnv/ tests/
uv lock --check
BPC_RUN_SLOW_TESTS=1 pytest tests/ --cov=BalloonPoppingGymEnv
```

`make format` fixes import order and formatting in place.

`uv lock --check` matters more than it looks. CI installs with pip, so nothing
else notices `uv.lock` drifting away from the ActiveRocketPy submodule.

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
change. A real one shows up in the leading digits, or in `popped_count`, or in
the step count.

A baseline is tied to the pair of (repository, submodule commit). Regenerating
one because it went red, without understanding why, defeats the point of having
it.

## Reporting issues

Use the templates. A bug report that names the scenario, the seed and the
ActiveRocketPy commit is one somebody can reproduce; without them it usually
turns into a round trip.

## Changes worth a changelog entry

Anything a competitor would notice goes in `CHANGELOG.md` under `Unreleased`:
environment behaviour, the action or observation space, the submission format,
scoring, or the supported Python versions.
