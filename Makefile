# Same paths and the same ruff CI uses. Covering only the package left `make
# format` reporting success while `ruff format --check BalloonPoppingGymEnv/
# tests/ doc/examples/` still failed, and a bare `ruff` picks up whatever is on
# PATH, which is how a locally formatted file can still fail CI.
.PHONY: format

format:
	@uvx ruff@0.15.20 check --select I --fix BalloonPoppingGymEnv/ tests/ doc/examples/
	@uvx ruff@0.15.20 format BalloonPoppingGymEnv/ tests/ doc/examples/
	@echo Ruff formatting completed.
