# Balloon Popping Challenge: A 6-DoF Rocket GNC Simulation [Gymnasium](https://gymnasium.farama.org/) Environment

<a target="_blank" href="https://colab.research.google.com/github/ARRC-Rocket/BalloonPoppingChallenge/blob/main/doc/examples/evaluate_scenario_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This repository contains the code for the Balloon Popping Challenge, a 6-DoF rocket guidance, navigation, and control (GNC) simulation environment built using [Gymnasium](https://gymnasium.farama.org/). The environment is designed to simulate an active controlled rocket to pop balloons scattered in the sky. The simulator incorporates realistic physics, including atmospheric conditions and rocket dynamics, to provide a challenging platform for developing and testing GNC algorithms. This project is based on [ActiveRocketPy](https://github.com/ARRC-Rocket/ActiveRocketPy), a fork of open-source software [RocketPy](https://github.com/RocketPy/RocketPy).

## Installation

Clone the repository and initialize the `ActiveRocketPy` submodule:

```shell
git clone https://github.com/ARRC-Rocket/BalloonPoppingChallenge.git
cd BalloonPoppingChallenge
git submodule update --init
```

Then set up the environment with **uv** (recommended) or with **pip**.

### Option A: uv (recommended)

[uv](https://docs.astral.sh/uv/) installs the pinned Python version, creates the virtual environment, and installs the locked dependencies in one step.

**1. Install uv** (skip if it is already installed).

Windows (PowerShell):

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.11.14/install.ps1 | iex"
```

macOS / Linux:

```shell
curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh
```

Other install methods (Homebrew, winget, pipx, ...) are in the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

**2. Open a new terminal** so the shell picks up `uv`, then confirm it works:

```shell
uv --version
```

**3. Set up the environment** from the repository root:

```shell
uv sync
```

Prefix any command with `uv run` to run it inside the environment, e.g. `uv run python -m unittest discover tests`.

> **`uv` not found?** First open a new terminal: a fresh shell is needed after installing. If it is still not found, uv's install directory is not on your `PATH`. Either add it to `PATH`, or install uv with `pip install uv==0.11.14` and use `python -m uv` in place of `uv` (for example `python -m uv sync`); `python -m uv ...` is equivalent to `uv ...` and does not depend on `PATH`.

### Option B: pip

```shell
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On Unix or macOS
python -m pip install -r requirements.txt
```

> The `vpython` renderer (`render_mode="vpython"`) is optional and is not installed by either option above; the default `matplotlib` renderer works without it. To enable it, install the `vpython` extra: `uv sync --extra vpython` (uv) or `python -m pip install -e ".[vpython]"` (pip).

## Update from the repository

```shell
cd BalloonPoppingChallenge
git pull origin main
git submodule update --remote --merge
uv sync   # re-sync the environment (pip users: python -m pip install -r requirements.txt)
```

## Examples

1. Evaluate agent:
    - Develop the agent in [/agents folder](./BalloonPoppingGymEnv/agents) by inheriting from [BaseAgent](./BalloonPoppingGymEnv/agents/base_agent.py) and implementing the `get_action` method.
    - Update the evaluation configuration file in [/evaluation/configs folder](./BalloonPoppingGymEnv/evaluation/configs) to specify the scenario parameters and the agent to be evaluated.
    - Run:

        ```shell
        cd BalloonPoppingChallenge
        python .\BalloonPoppingGymEnv\evaluation\evaluate.py .\BalloonPoppingGymEnv\evaluation\configs\example_eval_cfg.yaml
        ```

    - You should see a rocket popping static balloons in the sky:
        ![screen shot of scenario 0 running](doc/figures/scenario_0_screenshot.png)

2. Example code for development and debugging:
    - Run the example script:

        ```shell
        cd BalloonPoppingChallenge
        python .\doc\examples\run_env_agent.py
        ```

    - This will run the specified scenario with the example agent and print the final reward. You can modify the agent, scenario parameters, and other settings in the script for development and debugging purposes.

3. Example code for state estimation:
    - Run the example script:

        ```shell
        cd BalloonPoppingChallenge
        python .\doc\examples\test_navigation_agent.py
        ```

    - This will run the specified scenario with the example navigation agent. The comparison between the estimated and ground truth attitude and velocity is plotted.

4. Colab notebook example:
    - Open the [evaluate_scenario_colab.ipynb](./doc/examples/evaluate_scenario_colab.ipynb) notebook in Google Colab.
    - Follow the instructions in the notebook to run the evaluation in the cloud.

## Testing

Run the cleanup invariant tests (uses only the Python standard library and PyYAML):

```shell
python -m unittest discover tests
```

## Modelling Details

- Rocket flight modelling (RocketPy):
  - The details can be found in the [RocketPy Reference](https://docs.rocketpy.org/en/latest/index.html)
  - The coordinates are shown in the figure below:

    ![Rocket coordinate frames](doc/figures/Coordinates.drawio.svg)
- Balloon popping specific modelling:
  - Balloons are modeled as rigid spheres with a certain radius and mass.
  - Balloon flights are simulated using [Monte-Carlo simulation](./ActiveRocketPy/rocketpy/simulation/monte_carlo.py) method provided by ActiveRocketPy. The stochastic parameters are listed in the parameter files. A small solid motor will push the balloon out-of rail to start simulation of [Flight](./ActiveRocketPy/rocketpy/simulation/flight.py) class in ActiveRocketPy. The balloon will then fly freely under the influence of gravity, buoyancy, wind, and atmospheric drag.
  - The flight of each balloon is not affected by the rocket or other balloons.
  - A balloon is considered popped if the distance between the path of the rocket (center of dry mass) and the center of balloon within a timestep is less than the radius of the balloon (radius as given in the parameter file, not the stochastic value for monte-carlo simulation).
  - Balloons release will be determined depends on the scenario parameters.
  - There will be a single launch, and the aim is to pop as many balloons as possible.
  - Launch time, inclination, and heading are determined by the agent.
  - There will be disturbances, e.g., sensor noise, wind in the environment.

## Gymnasium Environment Operation

There are three stages in the operation of the Gymnasium environment: reset, stepping, and termination.

1. **Reset**: The environment is reset using `env.reset()`, which sets up the initial conditions for the rocket and balloons as given in the [scenario_0_parameter.yaml](./BalloonPoppingGymEnv/envs/scenario_parameters/scenario_0_parameters.yaml) files. The trajectory of each balloon is simulated using the [monte-carlo simulation of ActiveRocketPy](./ActiveRocketPy/rocketpy/simulation/monte_carlo.py) then stored in the environment.

2. **Stepping**: The agent takes an action (e.g., launch, roll, throttle and TVC commands) and calls `env.step(action)`, which advances the simulation by one time step. The environment returns the new observations, reward, termination flag, and additional info.

3. **Termination**: The episode ends when maximum simulation time is reached or the rocket hits the ground.

The actions, observations, info, reward in this environment are:

- actions:
  - `launch`: a binary command to launch the rocket.
  - `launch_inclination_heading`: a 2-element array [inclination, heading] representing the launch inclination (0-90 degrees from horizontal) and heading angles (0-360 degrees from north).
  - `tvc`: a 2-element array [TVC_x, TVC_y] representing the thrust vector control (TVC) gimbal angles (deg). Polarity: positive gimbal angles provide positive torques.
  - `throttle`: a scalar representing the throttle ratio between 0 and 1.
  - `roll`: a scalar representing the roll torque command in N-m.
- observations:
  - `simulation_time`: the current simulation time in seconds.
  - `balloon_status`: a n-element array representing the status of each balloon (0: on the ground, 1: released, 2: popped). n is the number of balloons in the scenario.
  - `balloon_states`: a n x 6 array representing the position (posX, posY, posZ) and velocity (velX, velY, velZ) of each balloon.
    - Position is the center of the balloon in the launch frame (relative to launch origin) in meters.
    - Velocity is the center of the balloon in the launch frame (relative to launch origin) in m/s.
  - `rocket_sensors`: a 12-element array representing the rocket's sensor measurements (gyroX, gyroY, gyroZ, accX, accY, accZ, posX, posY, posZ, velX, velY, velZ). Orientation of inertial sensors matches body frame. The measurements will be nan before launch action.
    - Gyroscopes measure the angular velocity (rad/s) in the rocket body frame.
    - Accelerometers measure the linear acceleration (m/s²) in the rocket body frame. Gravity is included in the accelerometer measurements.
    - GNSS sensors measure the position (m) and velocity (m/s) in the launch frame (relative to launch origin).
  - Note that the rocket's true states (e.g., attitude, angular velocity) are not directly observed by the agent, and the agent needs to infer them from the sensor measurements.
- info:
  - `rocket_states`: a 13-element array representing the rocket's true states. These states are not observed and should not be used by the agent but can be used for development and debugging. The states are [posX, posY, posZ, velX, velY, velZ, e0, e1, e2, e3, wX, wY, wZ]:
    - pos: center of dry mass position (m) in the launch frame (relative to launch origin).
    - vel: center of dry mass velocity (m/s) in the launch frame (relative to launch origin).
    - e: quaternion representing the attitude of the rocket (e0, e1, e2, e3) relative to the launch frame.
    - w: angular velocity (rad/s) in the rocket body frame.
  - `popped_count`: total number of balloon popped. This will be the final score of evaluation.

- reward:
  - The reward is the number of balloons popped at each time step.

## Known Limitations

- The mass properties are pre-calculated before flight according to max flow rate and burn time. Throttle commands does not affect the change of the mass properties in-flight. It is equivalent to throttling the Isp of rocket engine while the flow rate remains constant. The engine is cut off when burn time is reached

## Agent Development

Agents for evaluation are placed in the [/agents folder](./BalloonPoppingGymEnv/agents). They should be implemented as a class that inherits from [BaseAgent](./BalloonPoppingGymEnv/agents/base_agent.py) and implements the `get_action` method. The agent can access the scenario parameters through `self.given_parameters`, as defined in `scenario_given_parameters.yaml` files in [/scenario_parameters folder](./BalloonPoppingGymEnv/envs/scenario_parameters/). Observations are passed through the `get_action` method. The agent should output an action dictionary that matches the action space defined in the environment.

## Evaluation details

The evaluation script is located in [/evaluation folder](./BalloonPoppingGymEnv/evaluation). It takes a configuration file as input, which specifies the scenario parameters and the agent to be evaluated. The script runs the specified scenario with the given agent and outputs the results.

![flow chart of evaluation process](doc/figures/EvaluationFlowChart.drawio.svg)

## Reference

- [RocketPy GitHub](https://github.com/RocketPy/RocketPy)
- [RocketPy Documentation](https://docs.rocketpy.org/en/latest/index.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Citation

If you run Balloon Popping Challenge in your research, please consider citing:

```bibtex
@misc{BalloonPoppingChallenge,
  author = {Zuo-Ren Chen and Advanced Rocket Research Center (ARRC)},
  title = {Balloon Popping Challenge: A 6-DoF Rocket GNC Simulation Gymnasium Environment},
  month = {April},
  year = {2026},
  url = {https://github.com/ARRC-Rocket/BalloonPoppingChallenge}
}
```

___
___

## Balloon Popping Challenge @ TASTI 2026

*An International Rocket GNC Software Design Competition*

**Develop your own Python software to guide, navigate, and control a rocket to pop balloons in the sky.**

This is the official code repository for the Balloon Popping Challenge, a competition held at the Taiwan International Assembly of Space Science, Technology, and Industry (TASTI) 2026.

This competition aims to cultivate next-generation talent in rocket Guidance, Navigation, and Control (GNC) by engaging participants in the development of autonomous flight control algorithms.
Participants are required to develop Python-based software to autonomously control a simulated rocket. Within a physics-based simulation environment, the rocket must navigate and pop balloons released dynamically to maximize the number of pops under uncertain conditions.

Keywords: GNC, autonomous rocket, optimization, path-finding.

### Competition Details

- Sign up for the competition: `[TASTI 2026 Registration]()`
- Competition timeline:
  - **Apr dd, 2026**: Competition announcement, open applications, beta release of rules and software
  - **May - Aug, 2026**: Release software updates, update rules, hold monthly meetings, online leader boards
  - **Aug dd, 2026**: Release final software and rules, close applications
  - **Sep dd, 2026**: Online elimination rounds
  - **Oct dd, 2026**: Announce finalists
  - **Nov dd, 2026 @ TASTI**: Finalist presentations and live demos (<2 hours total)

### Competition Rules

- The participant will develop agents in [/agents folder](./BalloonPoppingGymEnv/agents/) to control a rocket.
- The agent will be initialized with the given paramter of each scenario.
- At each time step, the agent should only take the observations provided by the environment to output control commands (e.g., launch, roll, throttle and TVC commands). The agent should not have access to any other information about the environment or the simulator.
- Other than the agent, all other components of the simulator are fixed and provided by the organizer. Participants are not allowed to modify any other part of the codebase for the evaluation.
- Questions about the rules and software can be asked in the [GitHub Issues](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/issues). The organizer will hold regular meetings to answer questions and provide updates.
- Code suggestions, contributions, and bug reports to the codebase are highly welcomed. Please submit a pull request or open an issue for discussion.

### Competition Leaderboard

Submit your results to: [https://balloonpoppingchallenge.arrcrocket.org/](https://balloonpoppingchallenge.arrcrocket.org/).
To generate the required .pkl file for submission, please follow these steps:

1. Register for the competition [with this form](https://docs.google.com/forms/d/e/1FAIpQLSegCqnI4t-R_6Nxtbkf-XJ-V3L5-_DlyDxmSU_FY2Qa1lvLXQ/viewform). Organizors will send the `team_name` and `team_secret` through email.
2. Edit [eval_cfg.yaml](./BalloonPoppingGymEnv/evaluation/configs/example_eval_cfg.yaml)

    ```yaml
    team_name: example
    team_secret: 3b4b84252bc53eb1f4d8ea008a9243040088e71a1f1fd7a9ccfe203f9c9cb164
    leaderboard_submission: true
    ```

3. Run

    ```bash
    python .\BalloonPoppingGymEnv\evaluation\evaluate.py .\BalloonPoppingGymEnv\evaluation\configs\{eval_cfg}.yaml
    ```

4. Upload your .pkl file generated in [/results](BalloonPoppingGymEnv\evaluation\results) folder to the [leaderboard](https://balloonpoppingchallenge.arrcrocket.org/)

### Competition Scenarios

Exact scenario for elimination rounds and final rounds will be announced later. Below are some examples of possible scenarios.

|# | Name | 🚀 Actuator Response | 🚀 Sensor Noise | 🌬️ Wind | 🎈 Number | 🎈 Release Interval (sec) | 🎈 Initial Position | 🎈 Position Observation | 🎈 Velocity Observation |
|---|---|---|---|---|---|---|---|---|---|
|#0         |Hello World       |Ideal       |No             |None                  |10     |N/A    |height = arange(10, 410, 40) + elevation| Static at initial position           | no velocity                        |
|#1         |Random Balloon    |Ideal       |No             |Yes                   |100    |Random |Random at ground            |Full observation at current step      |Full observation at current step    |
|#2 (TBD)   |Noisy Sensor      |Ideal       |Yes            |Yes                   |100    |Random |Random at ground            |Full observation at current step      |Full observation at current step    |
|#3 (TBD)   |Clumsy Actuator   |LPF, random |Yes            |Yes                   |100    |Random |Random at ground            |Full observation at current step      |Full observation at current step    |
|#4 (TBD)   |Bad Weather       |LPF, random |Yes            |Yes, random magnitude |100    |Random |Random at ground            |Full observation at current step      |Full observation at current step    |
|#5 (TBD)   |Sensor Drop off   |LPF, random |Yes & drop-off |Yes, random magnitude |100    |Random |Random at ground            |Full observation at current step      |Full observation at current step    |
|#6 (TBD)   |Find the Balloon  |LPF, random |Yes & drop-off |Yes, random magnitude |100    |Random |Random at ground            |Partial observation at current step   |Partial observation at current step |
|#7 (TBD)   |Balloon Recovery

### Release Notes

- **v0.0.1**: Initial release of the codebase and rules.
- **v0.0.2**:
  - Fix/maintain:
    - Fix json file read encoding issue Fix evaluate's encode problem #5
    - Make vpython an optional, lazily-imported render dependency (Make vpython an optional, lazily-imported render dependency #15)
    - Clean up dead code, redundant calls, and unused imports (Clean up dead code, redundant calls, and unused imports #7)
    - Move Monte Carlo output out of the package directory (Move Monte Carlo output out of the package directory #22)
    - Add GitHub Actions CI workflow (#23)
  - Update:
    - Update to return reward from each step only. Total count of popped balloon is returned in info. Reward returned every step is the cumulative pop count, not the per-step delta #10 (ENH: Return reward of each step #12)
    - Skip Monte Carlo balloon simulation for scenario 0 (Skip Monte Carlo balloon simulation for scenario 0 #18)
    - Render every balloon in the vpython renderer (#27)
  - New:
    - Add uv as the recommended environment setup path (Add uv as the recommended environment setup path #20)
    - Add standardized issue templates (Add standardized issue templates #25)
    - ENH: Add architecture for post sim graphics (#28)
    - ENH: Add leaderboard submission functions (#31)
    - Add gust to balloon_world environment (#32)
