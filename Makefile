# Same paths CI checks. Covering only the package left `make format` reporting
# success while `ruff format --check BalloonPoppingGymEnv/ tests/ doc/examples/` still failed.
format:
	@ruff check --select I --fix BalloonPoppingGymEnv/ tests/ doc/examples/
	@ruff format BalloonPoppingGymEnv/ tests/ doc/examples/
	@echo Ruff formatting completed.