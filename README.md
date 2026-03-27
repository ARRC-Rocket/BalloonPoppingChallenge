# Balloon Popping Challenge: A 6-DoF Rocket GNC Simulation [Gymnasium](https://gymnasium.farama.org/) Environment

This repository contains the code for the Balloon Popping Challenge, a 6-DoF rocket guidance, navigation, and control (GNC) simulation environment built using [Gymnasium](https://gymnasium.farama.org/). The environment is designed to simulate an active controlled rocket to pop balloons scattered in the sky. The simulator incorporates realistic physics, including atmospheric conditions and rocket dynamics, to provide a challenging platform for developing and testing GNC algorithms. This project is based on [ActiveRocketPy](https://github.com/ARRC-Rocket/ActiveRocketPy), a fork of open-source software [RocketPy](https://github.com/RocketPy/RocketPy). 

## Installation

```bash
git clone https://github.com/ARRC-Rocket/BalloonPoppingChallenge.git
git submodule update --init # Initialize the ActiveRocketPy submodule
cd BalloonPoppingChallenge
python -m venv .venv        # Create a virtual environment (optional but recommended)
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Example

> WIP

```python
python example.py
```
## Reference
- [RocketPy GitHub](https://github.com/RocketPy/RocketPy)
- [RocketPy Documentation](https://docs.rocketpy.org/en/latest/index.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

___
___

## Balloon Popping Challenge @ TASTI 2026

*An International Rocket GNC Software Design Competition*

__Develop your own Python software to guide, navigate, and control a rocket to pop balloons in the sky.__

This is the official code repository for the Balloon Popping Challenge, a competition held at the Taiwan International Assembly of Space Science, Technology, and Industry (TASTI) 2026. The challenge tasks participants with developing guidance, navigation, and control (GNC) algorithms in a Python simulator to pop scattered balloons using a TVC-equipped throttleable rocket in the sky. All techniques are encouraged, including classic/modern/optimal control, as well as machine/reinforcement learning.

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

- The participant will develop agents to control a rocket in `[agent.py]()`.
- Many balloons will be released at random intervals and random locations around the rocket launch site.
- There will be a single launch, and the aim is to pop as many balloons as possible.
- Launch time is determined by the agents.
- There will be disturbances, e.g., sensor noise, wind in the environment.

### Competetion Scenarios
Exact scenario for elimation rounds and final rounds will be announced later. Below are some examples of possible scenarios.


|# | Name | 🚀 Throttle Range (TWR) | 🚀 TVC & Throttle Actuator Response | 🚀 Sensor Noise | 🌬️ Wind | 🎈 Number | 🎈 Release Interval (sec) | 🎈 Release Sequence | 🎈 Initial Position | 🎈 Position Observation | 🎈 Velocity Observation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| #0 | Hello World | 2, TBC | Ideal | No | None | 10 | N/A | N/A | height = linspace(10,100,10) | Static at initial position | Static at initial position, no velocity |
| #1 | Ideal World | [1, 2 (TBC)] | Ideal | No | None | 100 | 1, TBC | One by one | Random at ground | Free flight after release; Full observation at current step | Free flight after release; Full observation at current step |
| #2 | Random Balloon | [1, 2 (TBC)] | Ideal | No | None |  100 | Random | Random | Random at ground | Free flight after release; Full observation at current step | Free flight after release; Full observation at current step |
| #3 | Noisy Sensor | [1, 2 (TBC)] | Ideal | Yes, random magnitude | None | 100 | Random | Random | Random at ground | Free flight after release; Full observation at current step | Free flight after release; Full observation at current step |
| #4 | Clumsy Actuator | [1, 2 (TBC)] | LPF, random | Yes, random magnitude | None |100 | Random | Random | Random at ground | Free flight after release; Full observation at current step | Free flight after release; Full observation at current step |
| #5 | Bad Weather | [1, 2 (TBC)] | LPF, random | Yes, random magnitude | Yes, random magnitude | 100 | Random | Random | Random at ground | Free flight after release; Full observation at current step | Free flight after release; Full observation at current step |
| #6 | Time for Hovering | [0.5, 2] (TBC) | LPF, random | Yes, random magnitude | Yes, random magnitude | 100 | Random | Random | Random at ground | Free flight after release; Full observation at current step | Free flight after release; Full observation at current step |
| #7 | Sensor Drop off | [0.5, 2] (TBC) | LPF, random | Yes, random magnitude & drop-off | Yes, random magnitude | 100 | Random | Random | Random at ground | Free flight after release; Full observation at current step | Free flight after release; Full observation at current step |
| #8 | Find the Balloon | [0.5, 2] (TBC) | LPF, random | Yes, random magnitude & drop-off | Yes, random magnitude | 100 | Random | Random | Random at ground | Free flight after release; Partial observation at current step | Free flight after release; Partial observation at current step |

