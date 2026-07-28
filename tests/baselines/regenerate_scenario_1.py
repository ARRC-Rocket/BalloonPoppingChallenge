"""Regenerate the scenario #1 regression baseline (issue #38).

Run from the repository root *after a deliberate physics change* (and review the
resulting diff before committing):

    PYTHONPATH=. python tests/baselines/regenerate_scenario_1.py

Scenario #1 runs a 100-flight Monte Carlo balloon ensemble on reset, so this
takes ~30 s. The baseline is tied to the current repository + ActiveRocketPy
submodule state, so it must be regenerated whenever the rocket or balloon
physics is intentionally updated.
"""

from pathlib import Path

from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

from tests.baselines.baseline_io import write_baseline

from tests.test_scenario1_regression import (
    AGENT_KWARGS,
    BALLOON_INDEX_STRIDE,
    BALLOON_TIME_STRIDE,
    ROCKET_DOWNSAMPLE_STRIDE,
    SCENARIO_NUMBER,
    downsample_balloon_positions,
    post_launch_rocket_positions,
    run_scenario_1,
)

OUTPUT_PATH = Path(__file__).parent / "scenario_1.json"


def main():
    scenario_params, _ = load_scenario_parameters(SCENARIO_NUMBER)
    rocket_positions, balloon_positions, popped = run_scenario_1()
    baseline = {
        "scenario_number": SCENARIO_NUMBER,
        "random_seed": scenario_params["scenario"]["random_seed"],
        "agent": "AttitudeRateControlAgent",
        "agent_kwargs": AGENT_KWARGS,
        "rocket_downsample_stride": ROCKET_DOWNSAMPLE_STRIDE,
        "balloon_time_stride": BALLOON_TIME_STRIDE,
        "balloon_index_stride": BALLOON_INDEX_STRIDE,
        "num_steps_full": int(rocket_positions.shape[0]),
        "popped_count": int(popped),
        "rocket_position_downsampled": post_launch_rocket_positions(
            rocket_positions
        ).tolist(),
        "balloon_position_downsampled": downsample_balloon_positions(
            balloon_positions
        ).tolist(),
    }
    write_baseline(baseline, OUTPUT_PATH)
    rocket_rows = len(baseline["rocket_position_downsampled"])
    balloon_shape = (
        len(baseline["balloon_position_downsampled"]),
        len(baseline["balloon_position_downsampled"][0]),
    )
    print(
        f"wrote {OUTPUT_PATH} "
        f"(steps={baseline['num_steps_full']}, popped={baseline['popped_count']}, "
        f"rocket_rows={rocket_rows}, balloon_grid={balloon_shape})"
    )


if __name__ == "__main__":
    main()
