# aero-envs

Gymnasium-compatible reinforcement learning environments for the **Quanser Aero 2** — a twin-rotor aerodynamics testbed. The package provides three environment backends that share a common interface:

| Environment | Backend | Use case |
|---|---|---|
| `AeroSimulationEnv` | C shared library (`.dll` / `.so`) | Fast, offline RL training |
| `AeroHilEnv` | Quanser HIL SDK (real hardware) | Hardware-in-the-loop testing |
| `AeroRealEnv` | *(planned)* | Direct hardware deployment |

---

## Installation

**Requirements:** Python ≥ 3.8, Windows or Linux.

```bash
git clone https://github.com/JRC-ISIA/aero-envs.git
cd aero-envs
pip install -e .
```

Core dependencies are installed automatically:

```
gymnasium >= 0.26.0
numpy    >= 1.21.0
pygame   >= 2.0.0
```

For evaluation utilities and training baselines, also install:

```bash
pip install -r requirements.txt   # adds matplotlib, stable-baselines3, pytest
```

> [!NOTE]
> The hardware-in-the-loop environment (`AeroHilEnv`) additionally requires the **Quanser Hardware Interface Library** (`quanser`). This proprietary SDK is not available through PyPI and must be installed separately from Quanser.

---

## Environments

### `AeroSimulationEnv`

Simulates the Quanser Aero 2 pitch dynamics via a pre-compiled model (`.dll` on Windows, `.so` on Linux) exposed through Python's `ctypes`. It runs faster than real-time and supports rendering via Pygame.

**Gymnasium ID:** `AeroSimulationEnv-v0`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `step_size` | `float` | `0.02` | Internal simulation step size (s). Must evenly divide `sample_time`. |
| `stop_time` | `float` | `15.0` | Episode duration (s). |
| `sample_time` | `float` | `0.1` | Agent interaction period (s). |
| `target_tilt` | `float \| TrajectoryGenerator` | `RandomStepTrajectory()` | Target pitch angle (rad) or a trajectory object. |
| `initial_tilt` | `float \| Callable` | `0` | Initial pitch angle (rad) or a callable that returns one. |
| `render_mode` | `str \| None` | `None` | `"human"`, `"rgb_array"`, or `None`. |
| `norm_action` | `bool` | `False` | Normalize actions to `[-1, 1]` (maps internally to `[-24, 24]` V). |
| `norm_observation` | `bool` | `False` | Normalize observations to `[-1, 1]`. |
| `power_penalty_weight` | `float` | `0.0` | Weight `α` of the power-consumption penalty in the reward. |

### `AeroHilEnv`

Wraps the same base interface but talks to a real Quanser Aero 2 over the HIL SDK. It enforces real-time timing by busy-waiting until each `sample_time` period elapses.

**Gymnasium ID:** `AeroHilEnv-v0`

Parameters are identical to `AeroSimulationEnv` except there is no `step_size` or `render_mode`. An additional `card_identifier` parameter specifies the HIL board connection string (defaults to `"0@tcpip://localhost:18950?..."`).

---

## Observation & Action Spaces

**Observation** (`gym.spaces.Dict`):

| Key | Shape | Range (unnormalized) |
|---|---|---|
| `pitch` | `(1,)` | `[-π/2, π/2]` rad |
| `velocity` | `(1,)` | `[-0.244, 0.244]` rad/step |
| `target` | `(1,)` | `[-π/2, π/2]` rad |

**Action** (`gym.spaces.Box`, shape `(1,)`):
- Unnormalized: voltage in `[-24, 24]` V applied to the two rotors (equal and opposite).
- Normalized (`norm_action=True`): `[-1, 1]`, mapped to `[-24, 24]` V internally.

**Reward:**

$$r = (1 - \alpha)\,(-|\theta_\text{target} - \theta|) + \alpha\,\left(-\frac{P}{P_\text{max}}\right)$$

where $\alpha$ is `power_penalty_weight`, $\theta$ is the current pitch, and $P_\text{max} = 24 \times 3.9197$ W. The episode truncates (with a scaled penalty) if $|\theta| \geq \pi/2$.

---

## Quick Start

### Random-action rollout

```python
import gymnasium as gym
import aero_envs  # registers the environments

env = gym.make("AeroSimulationEnv-v0", render_mode="human")
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

### Custom trajectory with observation wrappers

```python
import gymnasium as gym
import numpy as np
import aero_envs
from aero_envs.utils.trajectories import TrajectoryGenerator
from aero_envs.wrappers.observation_wrappers import FutureTargetsWrapper

def build_trajectory() -> TrajectoryGenerator:
    traj = TrajectoryGenerator()
    traj.add_ramp(-20 * np.pi / 180, 10.0)
    traj.add_sine(amplitude=15 * np.pi / 180, frequency=0.05, duration=10.0)
    traj.add_ramp(0, 5.0)
    traj.add_step(35 * np.pi / 180, 10.0)
    traj.add_s_curve(0, 10.0)
    traj.add_hold(5.0)
    return traj

trajectory = build_trajectory()
trajectory.plot()  # visualise with matplotlib

env = gym.make(
    "AeroSimulationEnv-v0",
    render_mode="human",
    target_tilt=trajectory,
    stop_time=trajectory.total_duration,
    norm_action=True,
    norm_observation=True,
    initial_tilt=lambda: np.random.uniform(-40 * np.pi / 180, 40 * np.pi / 180),
)
env = FutureTargetsWrapper(env, num_future_targets=4, future_target_interval=0.5)

obs, info = env.reset()
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step([0.0])
    done = terminated or truncated
env.close()
```

### Evaluating a trained policy

```python
from stable_baselines3 import PPO
from aero_envs.utils.evaluation import evaluate_agent
import numpy as np

model = PPO.load("examples/example_models/ppo_example")
results = evaluate_agent(model, plot=True, num_episodes=10, stop_time=200.0, seed=42)

print(f"Deviation : {np.mean(results['deviation']):.4f} ± {np.std(results['deviation']):.4f} rad")
print(f"Power     : {np.mean(results['power']):.4f} ± {np.std(results['power']):.4f} W")
print(f"Action    : {np.mean(results['action']):.4f} ± {np.std(results['action']):.4f} V")
```

---

## Trajectory API

The `TrajectoryGenerator` class lets you compose piecewise reference signals. Segments are evaluated efficiently with binary search (O(log N)).

```python
from aero_envs.utils.trajectories import TrajectoryGenerator
import numpy as np

traj = TrajectoryGenerator()
traj.add_hold(3.0)                                    # stay at 0 rad for 3 s
traj.add_s_curve(40 * np.pi / 180, 5.0)              # smooth ramp to +40° over 5 s
traj.add_ramp(-20 * np.pi / 180, 4.0)                # linear ramp to -20°
traj.add_sine(amplitude=15 * np.pi / 180,
              frequency=0.1, duration=10.0)           # oscillate
traj.add_step(0.0, 5.0)                               # snap back to 0°

traj.plot()                                           # preview
value = traj(12.5)                                    # query at t = 12.5 s
```

Available segment types:

| Method | Description |
|---|---|
| `add_hold(duration)` | Hold the current value |
| `add_step(value, duration)` | Instant jump to `value` |
| `add_ramp(end_value, duration)` | Linear interpolation |
| `add_s_curve(end_value, duration)` | Cosine ease-in/out (recommended for hardware) |
| `add_sine(amplitude, frequency, duration)` | Sinusoidal oscillation |

Pre-built trajectories:

| Class | Description |
|---|---|
| `RandomTrajectory` | Random s-curve / hold segments, re-generated on each episode reset |
| `RandomStepTrajectory` | Random step targets; default for `AeroSimulationEnv` |
| `EvaluationTrajectory` | Fixed mixed trajectory for reproducible benchmarking |
| `EvaluationStepTrajectory` | Fixed step sequence from 0° to ±5°, ±20°, ±40° |

---

## Observation Wrappers

Located in [`src/aero_envs/wrappers/observation_wrappers.py`](src/aero_envs/wrappers/observation_wrappers.py):

| Wrapper | Added observation key | Description |
|---|---|---|
| `FutureTargetsWrapper` | `future_targets` | Appends N future target values sampled at a configurable time interval |
| `TargetVelocityWrapper` | `target_velocity` | Appends the current target trajectory derivative |
| `PowerPenaltyWrapper` | `power_penalty` | Exposes the current `power_penalty_weight` scalar |

Located in [`src/aero_envs/wrappers/wrappers.py`](src/aero_envs/wrappers/wrappers.py):

| Wrapper | Description |
|---|---|
| `PreferenceConditionedPowerPenaltyWrapper` | Randomises `power_penalty_weight` each episode and appends the value (`alpha`) to the observation; enables a single policy to cover the full accuracy–efficiency Pareto front |

---

## Running Tests

```bash
pytest tests/
```

The test suite covers environment initialisation, reset/step API conformance, observation / action space validation, normalisation, rendering, and gymnasium registration.

---

## Project Structure

```
aero-envs/
├── examples/
│   ├── basic_usage.py          # minimal random-action rollout
│   ├── advanced_usage.py       # custom trajectories + wrappers
│   ├── evaluation.py           # evaluate a saved SB3 policy
│   └── example_models/
│       └── ppo_example.zip     # pretrained PPO checkpoint
├── src/aero_envs/
│   ├── core/
│   │   ├── base_env.py         # AeroBaseEnv — shared gym.Env logic
│   │   └── aero_interface.py   # ctypes bindings & platform detection
│   ├── simulation/
│   │   └── simulation_env.py   # AeroSimulationEnv
│   ├── hil/
│   │   └── hil_env.py          # AeroHilEnv (Quanser HIL SDK)
│   ├── real/
│   │   └── real_env.py         # AeroRealEnv (planned)
│   ├── utils/
│   │   ├── trajectories.py     # TrajectoryGenerator + presets
│   │   └── evaluation.py       # run_episode / evaluate_agent helpers
│   ├── wrappers/
│   │   ├── observation_wrappers.py
│   │   └── wrappers.py
│   └── resources/
│       ├── windows/model.dll   # compiled Aero 2 simulation model
│       └── linux/model.so
├── tests/
│   └── test_simulation.py
├── requirements.txt
└── setup.py
```

---

## Platform Support

| Platform | Simulation | HIL |
|---|---|---|
| Windows | ✅ (`model.dll`) | ✅ |
| Linux | ✅ (`model.so`) | ✅ |
| macOS | ❌ | ❌ |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-65993-5_55,
  author    = {Sch{\"a}fer, Georg and Schirl, Max and Rehrl, Jakob and Huber, Stefan and Hirlaender, Simon},
  title     = {Python-Based Reinforcement Learning on Simulink Models},
  booktitle = {Combining, Modelling and Analyzing Imprecision, Randomness and Dependence},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  pages     = {449--456},
  isbn      = {978-3-031-65993-5}
}
```
