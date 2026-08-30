"""Internal setup helpers for the RocketPy CG/CP workshop notebook.

The notebook keeps user-facing controls and plots. This module owns the
mechanical work of loading a competition scenario and constructing RocketPy
objects so readers do not need to step through boilerplate model assembly.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from rocketpy import Environment, Rocket
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank


TAIWAN_TZ = ZoneInfo("Asia/Taipei")
DEFAULT_GFS_HOUR_TW = 2


@dataclass
class DemoContext:
    """Objects and derived values used by later notebook sections."""

    scenario_path: Path
    simulation_cfg: dict
    rocket_cfg: dict
    environment: Environment
    rocket: Rocket
    effective_body_cg_m: float
    run_started_tw: datetime
    launch_time_tw: datetime
    initial_margin_for_tag: float
    run_suffix: str
    output_dir: Path

    def output_path(self, stem: str, extension: str) -> Path:
        """Return one output path carrying this run's comparison suffix."""
        return self.output_dir / f"{stem}{self.run_suffix}{extension}"


def _resolve_forecast_time(
    forecast_date_tw: date | datetime | None,
    forecast_time_tw: datetime | None,
    run_started_tw: datetime,
) -> datetime:
    """Resolve a timezone-aware launch time for the atmospheric model."""
    if forecast_time_tw is not None:
        if forecast_time_tw.tzinfo is None:
            return forecast_time_tw.replace(tzinfo=TAIWAN_TZ)
        return forecast_time_tw.astimezone(TAIWAN_TZ)

    if forecast_date_tw is None:
        selected_date = run_started_tw.date()
    elif isinstance(forecast_date_tw, datetime):
        selected_date = (
            forecast_date_tw.astimezone(TAIWAN_TZ).date()
            if forecast_date_tw.tzinfo is not None
            else forecast_date_tw.date()
        )
    else:
        selected_date = forecast_date_tw

    # RocketPy's current GFS endpoint exposes the 02:00 Taiwan slot reliably;
    # using a fixed slot also keeps same-day static-margin runs comparable.
    return datetime(
        selected_date.year,
        selected_date.month,
        selected_date.day,
        DEFAULT_GFS_HOUR_TW,
        tzinfo=TAIWAN_TZ,
    )


def _build_environment(
    *,
    launch_time_tw: datetime,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float,
    weather_model: str,
    wind_profile_top_agl_m: float,
) -> Environment:
    environment = Environment(
        date=launch_time_tw,
        latitude=latitude_deg,
        longitude=longitude_deg,
        elevation=elevation_m,
        datum="WGS84",
        timezone="Asia/Taipei",
        max_expected_height=max(5000.0, wind_profile_top_agl_m),
    )
    normalized_model = weather_model.lower()
    if normalized_model == "gfs":
        try:
            environment.set_atmospheric_model(type="Forecast", file="GFS")
            # GFS may replace the pad elevation with coarse grid terrain.
            environment.set_elevation(elevation_m)
        except Exception as error:
            raise RuntimeError(
                "Could not load the requested GFS forecast. Check the internet "
                "connection or explicitly set WEATHER_MODEL to "
                "'standard_atmosphere' for the S0-aligned no-wind control."
            ) from error
    elif normalized_model == "standard_atmosphere":
        environment.set_atmospheric_model(type="standard_atmosphere")
    else:
        raise ValueError("WEATHER_MODEL must be 'GFS' or 'standard_atmosphere'.")
    return environment


def _build_hybrid_motor(cfg: dict) -> HybridMotor:
    tank_cfg = cfg["tank"]
    motor_cfg = cfg["motor"]
    oxidizer_tank = MassFlowRateBasedTank(
        name="oxidizer_tank",
        geometry=CylindricalTank(
            radius_function=tank_cfg["radius"], height=tank_cfg["height"]
        ),
        flux_time=(0, tank_cfg["flux_time"]),
        initial_liquid_mass=tank_cfg["initial_liquid_mass"],
        initial_gas_mass=tank_cfg["initial_gas_mass"],
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=tank_cfg["liquid_mass_flow_rate_out"],
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=Fluid(name=tank_cfg["liquid"], density=tank_cfg["liquid_density"]),
        gas=Fluid(name=tank_cfg["gas"], density=tank_cfg["gas_density"]),
    )
    motor = HybridMotor(
        thrust_source=motor_cfg["thrust_source"],
        dry_mass=motor_cfg["dry_mass"],
        dry_inertia=tuple(motor_cfg["dry_inertia"]),
        center_of_dry_mass_position=motor_cfg["center_of_dry_mass_position"],
        burn_time=(0, motor_cfg["burn_time"]),
        reshape_thrust_curve=False,
        grain_number=motor_cfg["grain_number"],
        grain_separation=motor_cfg["grain_separation"],
        grain_outer_radius=motor_cfg["grain_outer_radius"],
        grain_initial_inner_radius=motor_cfg["grain_initial_inner_radius"],
        grain_initial_height=motor_cfg["grain_initial_height"],
        grain_density=motor_cfg["grain_density"],
        nozzle_radius=motor_cfg["nozzle_radius"],
        throat_radius=motor_cfg["throat_radius"],
        interpolation_method="linear",
        nozzle_position=motor_cfg["nozzle_position"],
        grains_center_of_mass_position=motor_cfg["grains_center_of_mass_position"],
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    motor.add_tank(oxidizer_tank, position=tank_cfg["tank_position"])
    return motor


def _assemble_rocket(
    cfg: dict, motor: HybridMotor, body_cg_position: float
) -> Rocket:
    body = cfg["rocket_body"]
    rocket = Rocket(
        radius=body["radius"],
        mass=body["mass"],
        inertia=tuple(body["inertia"]),
        center_of_mass_without_motor=body_cg_position,
        power_off_drag=body["power_off_drag"],
        power_on_drag=body["power_on_drag"],
        coordinate_system_orientation="tail_to_nose",
        volume=body["volume"],
    )
    rocket.add_motor(motor, position=cfg["motor"]["motor_position"])
    rocket.add_nose(**cfg["nose"])
    fins = cfg["fins"]
    if fins["useFins"]:
        rocket.add_trapezoidal_fins(
            n=fins["n"],
            span=fins["span"],
            root_chord=fins["root_chord"],
            tip_chord=fins["tip_chord"],
            position=fins["position"],
        )
    return rocket


def _build_competition_rocket(
    cfg: dict, target_initial_gap_m: float | None
) -> tuple[Rocket, float]:
    motor = _build_hybrid_motor(cfg)
    body_cg = cfg["rocket_body"]["center_of_mass_without_motor"]
    rocket = _assemble_rocket(cfg, motor, body_cg)
    if target_initial_gap_m is not None:
        cp_at_ignition = float(rocket.cp_position(0))
        motor_mass = float(rocket.motor.total_mass(0))
        motor_cg = float(rocket.motor_center_of_mass_position(0))
        target_total_cg = cp_at_ignition + float(target_initial_gap_m)
        body_cg = (
            target_total_cg * (rocket.mass + motor_mass) - motor_cg * motor_mass
        ) / rocket.mass
        rocket = _assemble_rocket(cfg, motor, body_cg)
    return rocket, body_cg


def prepare_demo(
    *,
    repo_root: Path,
    scenario_number: int,
    body_cg_shift_m: float,
    fin_position_shift_m: float,
    target_initial_static_margin_cal: float | None,
    target_initial_cg_cp_gap_m: float | None,
    launch_latitude_deg: float,
    launch_longitude_deg: float,
    launch_elevation_m: float,
    weather_model: str,
    forecast_date_tw: date | datetime | None,
    forecast_time_tw: datetime | None,
    wind_profile_top_agl_m: float,
    output_dir: Path,
) -> DemoContext:
    """Load one scenario and construct the environment and competition rocket."""
    scenario_path = (
        repo_root
        / "BalloonPoppingGymEnv/envs/scenario_parameters"
        / f"scenario_{scenario_number}_parameters.yaml"
    )
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    with scenario_path.open(encoding="utf-8-sig") as stream:
        scenario = yaml.safe_load(stream)

    simulation_cfg = scenario["simulation"]
    rocket_cfg = deepcopy(scenario["rocket"])
    rocket_cfg["rocket_body"]["center_of_mass_without_motor"] += body_cg_shift_m
    rocket_cfg["fins"]["position"] += fin_position_shift_m

    if (
        target_initial_static_margin_cal is not None
        and target_initial_cg_cp_gap_m is not None
    ):
        raise ValueError(
            "Set only one of TARGET_INITIAL_STATIC_MARGIN_CAL and "
            "TARGET_INITIAL_CG_CP_GAP_M."
        )
    target_initial_gap_m = target_initial_cg_cp_gap_m
    if target_initial_static_margin_cal is not None:
        target_initial_gap_m = (
            float(target_initial_static_margin_cal)
            * 2
            * rocket_cfg["rocket_body"]["radius"]
        )

    run_started_tw = datetime.now(TAIWAN_TZ)
    launch_time_tw = _resolve_forecast_time(
        forecast_date_tw, forecast_time_tw, run_started_tw
    )
    environment = _build_environment(
        launch_time_tw=launch_time_tw,
        latitude_deg=launch_latitude_deg,
        longitude_deg=launch_longitude_deg,
        elevation_m=launch_elevation_m,
        weather_model=weather_model,
        wind_profile_top_agl_m=wind_profile_top_agl_m,
    )
    rocket, effective_body_cg_m = _build_competition_rocket(
        rocket_cfg, target_initial_gap_m
    )
    initial_margin_for_tag = float(rocket.static_margin(0))
    run_timestamp_tw = run_started_tw.strftime("%Y%m%d_%H%M%S")
    run_suffix = f"_{initial_margin_for_tag:.2f}cal_{run_timestamp_tw}"

    return DemoContext(
        scenario_path=scenario_path,
        simulation_cfg=simulation_cfg,
        rocket_cfg=rocket_cfg,
        environment=environment,
        rocket=rocket,
        effective_body_cg_m=effective_body_cg_m,
        run_started_tw=run_started_tw,
        launch_time_tw=launch_time_tw,
        initial_margin_for_tag=initial_margin_for_tag,
        run_suffix=run_suffix,
        output_dir=output_dir,
    )
