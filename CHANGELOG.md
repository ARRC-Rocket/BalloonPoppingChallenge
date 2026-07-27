# Changelog

Notable changes to BalloonPoppingGymEnv, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This tracks what a competitor would notice: environment behaviour, the action and
observation spaces, the submission format, scoring, and the supported Python
versions. Entries link to the pull request that made the change.

## [Unreleased]

Changes on `develop` since v0.0.2.

### Added

- Actuator dynamics: `gimbal_time_constant`, `roll_torque_time_constant` and
  `throttle_time_constant` give each actuator a first-order lag. `null` keeps the
  previous behaviour and remains the default in both shipped scenarios (#39).
- Rocket sensors are seeded from the scenario seed, so a run with a fixed seed
  reproduces its sensor noise (#54).
- Golden-master regression tests for scenarios 0 and 1, comparing the rocket and
  balloon trajectories against stored baselines so an ActiveRocketPy change
  cannot move a score unnoticed (#47, #49).
- Unit tests for the pop-detection geometry (#48) and for actuator dynamics
  (#66).
- CI checks that `uv.lock` is current against the ActiveRocketPy submodule (#62).

### Changed

- Diagnostics go through the `logging` module instead of `print` (#51).
- ActiveRocketPy updated to the RocketPy v1.13 line (#60), with `uv.lock`
  relocked to match (#46, #62).
- ruff is pinned in CI so formatting results are reproducible (#50).

### Fixed

- An agent that never sends a launch action now reaches the timeout and ends the
  episode instead of raising `AttributeError`, and the interval between launch
  and the first simulated sample takes part in pop detection rather than being
  skipped (#65).
- Submissions can be packed again: the sensor seeds are JSON-safe, where before
  `pack_for_submission` raised `TypeError` on a `SeedSequence` (#67).
- The `evaluate.py` integrity check no longer reports a mismatch for a file that
  differs only by a BOM or by line endings (#55).
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

[Unreleased]: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/compare/v0.0.2...develop
[0.0.2]: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.0.1
