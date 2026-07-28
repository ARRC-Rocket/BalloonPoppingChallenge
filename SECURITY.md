# Security

## Reporting a vulnerability

Report privately through GitHub, on the **Security** tab of this repository, using
**Report a vulnerability**. That opens a private advisory visible only to the
maintainers, so a competition-affecting problem does not become public before
there is a fix.

Please do not open a normal issue or pull request for something exploitable.
Issues here are public, and this repository is the code every competitor runs.

## What is in scope

This repository holds the simulation environment, the evaluation entry point and
the code that packs a submission. Anything that would let somebody read or alter
another team's submission, tamper with a score, or run code on a machine that
opens a submission, is in scope.

The leaderboard service is a separate, private repository. If a report concerns
the upload endpoint or the score storage rather than this code, say so in the
report and it will be routed there.

The physics engine is [ActiveRocketPy](https://github.com/ARRC-Rocket/ActiveRocketPy),
a fork of [RocketPy](https://github.com/RocketPy-Team/RocketPy). A vulnerability
that reproduces against upstream RocketPy belongs upstream; one that only affects
the fork belongs here.

## Which versions are covered

The most recent release on `main`. Older tags are not patched separately.

## What to expect

Reports are read by the maintainers. This project is run by a small team around a
competition schedule, so please allow for that rather than expecting a fixed
response window.
