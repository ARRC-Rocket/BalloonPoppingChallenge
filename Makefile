# Same paths CI checks. Covering only the package left `make format` reporting
# success while `ruff format --check BalloonPoppingGymEnv/ tests/` still failed.
format:
	@ruff check --select I --fix BalloonPoppingGymEnv/ tests/
	@ruff format BalloonPoppingGymEnv/ tests/
	@echo Ruff formatting completed.