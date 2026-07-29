# Changelog

Notable changes to BalloonPoppingGymEnv, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This tracks what users and contributors would notice: environment behaviour, the
action and observation spaces, the submission format, scoring, the supported
Python versions, and the tooling needed to work on the repository. Entries link
to the pull request that made the change.

## [Unreleased]

*Changes on `develop` since v0.1.0.*

## [0.1.0] - 2026-07-29

Released as [v0.1.0](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.1.0)
in #61, folding in the work from #39 through #111.

### Added

- Actuator dynamics: `gimbal_time_constant`, `roll_torque_time_constant` and
  `throttle_time_constant` give each actuator a first-order lag. `null` keeps the
  previous behaviour and remains the default in both shipped scenarios (#39).
- Rocket sensors are seeded from the scenario seed, so a run with a fixed seed
  reproduces its sensor noise (#54).
- Golden-master regression tests for scenarios 0 and 1, comparing the rocket and
  balloon trajectories against stored baselines so an ActiveRocketPy change
  cannot move a score unnoticed (#47, #49). Scenario 0's baseline also records
  the step each balloon is popped on and how the episode ends (#100).
- `scripts/verify_submission.py`, which regenerates a submission's balloons from
  the shipped scenario and its seed and reports anything that does not match.
  The balloons are not something an agent commands, so they are reproducible
  independently of the run (#97).
- Unit tests for the pop-detection geometry (#48) and for actuator dynamics
  (#66, #71).
- `CONTRIBUTING.md` and this changelog, with `make format` matching what CI
  checks (#76). `CITATION.cff` and `SECURITY.md`, and release notes grouped by
  kind (#79).
- The pop rule is written down in the README and pinned by tests, along with the
  coordinate frame observations are reported in (#81, #87).
- CI checks that `uv.lock` is current against the ActiveRocketPy submodule (#62)
  and runs on the Python floor the package claims rather than only asserting it
  (#75).
- Tests for the submission path (#77), scenario 1's release states (#83), the
  3D position error rather than each coordinate alone (#85), and the render
  behaviour a previous change had deleted (#86). The import guards no longer
  hide an installed but broken simulation stack (#82), and baselines are written
  as strict JSON, atomically, and refused when diverged (#84).

### Changed

- **Breaking, every agent loop:** running out of the precomputed horizon is
  reported as `truncated` rather than `terminated`. v0.0.2 returned
  `terminated=True` for both endings, so a loop written as
  `while not terminated:` never ends on the horizon: it keeps calling `step()`
  and the environment reads past the end of the balloon trajectories. Use
  `while not (terminated or truncated):` and keep the fourth value `step()`
  returns. `terminated` now means only that the flight ended. For scenario 1 the
  horizon is the usual ending (#94, #102, #104).
- **Breaking, custom scenarios:** `rocket.control.gimbal_range` is now
  `max_gimbal_angle`, and the three time-constant keys above are required. A
  scenario file written for v0.0.2 raises `KeyError: 'max_gimbal_angle'` on the
  first launch action. Rename the key and add the three fields as `null` to keep
  the previous behaviour (#39).
- The observation and the whitelisted `given_parameters` handed to an agent are
  copies. Writing into them used to reach the environment's own state, which was
  a scoring hole rather than an API detail (#99, #103).
- Diagnostics go through the `logging` module instead of `print` (#51).
- ActiveRocketPy updated to the RocketPy v1.13 line (#60), with `uv.lock`
  relocked to match (#62).
- ruff is pinned in CI so formatting results are reproducible (#50).
- The README's update instructions point at the pinned ActiveRocketPy commit
  rather than at whatever its branch currently holds (#72).

### Fixed

- The log line for running out of time said "Terminated", and a truncated
  episode drew no final frame unless its last step happened to land on the
  render cadence. Both read `terminated` from when that one flag also covered
  the clock (#104).
- A balloon Monte Carlo that returns fewer trajectories than were asked for is
  refused. A keyboard interrupt is caught upstream and returns what it has, and
  exactly one returned trajectory broadcast into a full hundred-balloon world in
  which every balloon flew the same path, scoring normally against a world that
  was never simulated (#101).
- The Monte Carlo workspace is removed after a successful run instead of leaving
  around 169 MB per run in the system temp directory, and kept when a run fails
  so there is something to read (#96).
- Two swept segments are classified as parallel by the angle between them rather
  than by how long they are. A millimetre-scale pair at ninety degrees was
  called parallel and reported 1.5005 m where the real distance is 1.4995 m,
  which against a 1.5 m balloon radius is a pop reported as a miss (#98).
- **Scoring:** the first interval after launch is checked for pops. The sweep
  started from the previous recorded position, which is all-NaN before launch,
  so the step where the rocket leaves the pad was skipped. It tracks the sweep
  origin explicitly now (#69).
- The submission file is written atomically and its name no longer carries the
  team name into the path. Two runs in the same second used to overwrite each
  other (#73).
- `balloon_flights` is no longer packed into a submission. Nothing reads it and
  `trajectories` already holds the same positions, so submissions are much
  smaller (#74).
- The console handler no longer competes with the level the CLI sets, so the
  score line prints once at the level asked for (#95).
- The documented coordinate frame was wrong about Z. X and Y are east and north
  of the launch point, but Z is altitude above sea level, not height above the
  pad, in `balloon_states`, the GNSS sensors and the rocket state alike. A
  balloon 10 m above the pad reads `elevation + 10`, and the rocket starts at
  `z = elevation` (#68).
- An agent that never sends a launch action now reaches the timeout and ends the
  episode instead of raising `AttributeError`, and the interval between launch
  and the first simulated sample takes part in pop detection rather than being
  skipped (#65).
- Submissions can be packed again: the sensor seeds are JSON-safe, where before
  `pack_for_submission` raised `TypeError` on a `SeedSequence` (#67).
- The `evaluate.py` integrity check no longer reports a mismatch for a file that
  differs only by a BOM, by line endings, or by a trailing newline (#55).
- A network failure during the `evaluate.py` check no longer discards a
  submission that has already been simulated (#68).
- Lint and pytest failures in the test suite (#40).

## [0.0.2] - 2026-05-30

Released as [v0.0.2](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.0.2)
in #33, folding in the work from #5 through #34.

### Added

- GitHub Actions CI, and a test suite to run in it (#23).
- Leaderboard submission packing (#31).
- Wind gust in the balloon world environment (#32).
- Post-simulation graphics architecture (#28), and the vpython renderer now draws
  every balloon (#27).
- Standardized issue templates (#25).
- `uv` as the recommended environment setup path (#20).

### Changed

- `step()` returns the reward for that step rather than the running total (#12).
- vpython became an optional, lazily imported dependency, so the default install
  no longer needs it (#15).
- Scenario 0 skips the Monte Carlo balloon simulation it never used (#18).
- Monte Carlo temporary output moved out of the package directory (#22).

### Fixed

- An encoding problem in `evaluate` (#5).
- Dead code, redundant calls and unused imports removed (#7).

## [0.0.1] - 2026-04-08

First public release: the Gymnasium environment, the example agents, scenarios 0
and 1, and the Colab example (#1, #3, #4).

[Unreleased]: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/compare/v0.1.0...develop
[0.1.0]: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.0.1
