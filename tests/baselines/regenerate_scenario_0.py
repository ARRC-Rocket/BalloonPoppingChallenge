"""Regenerate the scenario #0 regression baseline (issue #38).

Run from the repository root *after a deliberate physics change* (and review the
resulting diff before committing):

    PYTHONPATH=. python tests/baselines/regenerate_scenario_0.py

The baseline is tied to the current repository + ActiveRocketPy submodule state,
so it must be regenerated whenever the rocket physics is intentionally updated.
"""

from pathlib import Path

from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

from tests.baselines.baseline_io import write_baseline

from tests.test_scenario0_regression import (
    AGENT_KWARGS,
    DOWNSAMPLE_STRIDE,
    SCENARIO_NUMBER,
    post_launch_positions,
    run_scenario_0,
)

OUTPUT_PATH = Path(__file__).parent / "scenario_0.json"


def main():
    scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
    positions, popped = run_scenario_0()
    baseline = {
        "scenario_number": SCENARIO_NUMBER,
        "random_seed": scenario_params["scenario"]["random_seed"],
        "agent": "AttitudeRateControlAgent",
        "agent_kwargs": AGENT_KWARGS,
        "downsample_stride": DOWNSAMPLE_STRIDE,
        "num_steps_full": int(positions.shape[0]),
        "popped_count": int(popped),
        "rocket_position_downsampled": post_launch_positions(positions).tolist(),
    }
    write_baseline(baseline, OUTPUT_PATH)
    print(
        f"wrote {OUTPUT_PATH} "
        f"(steps={baseline['num_steps_full']}, popped={baseline['popped_count']})"
    )


if __name__ == "__main__":
    main()
