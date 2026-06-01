import matplotlib.pyplot as plt
from aero_envs.utils.trajectories import RandomTrajectory
from aero_envs.simulation.simulation_env import AeroSimulationEnv
import numpy as np
from gymnasium.wrappers import FlattenObservation


def run_episode(model, env) -> dict[str, list[float]]:
    """
    Run a single episode using the provided model and environment.
    """
    targets = []
    pitches = []
    powers = []
    actions = []
    obs, info = env.reset()
    done = False
    flatten_observation = isinstance(env, FlattenObservation)
    MAX_OBSERVATION_VALUES = env.env.MAX_OBSERVATION_VALUES if flatten_observation else env.MAX_OBSERVATION_VALUES
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        pitch = obs[0] if flatten_observation else obs["pitch"]
        target = obs[1] if flatten_observation else obs["target"]
        targets.append(target * MAX_OBSERVATION_VALUES["target"])
        pitches.append(pitch * MAX_OBSERVATION_VALUES["pitch"])
        powers.append(env.env.power if flatten_observation else env.power)
        actions.append(action[0] * 24)
        done = terminated or truncated

    return {
        "target": targets,
        "pitch": pitches,
        "power": powers,
        "action": actions,
    }

def plot_episode(data):
    time_seconds = np.arange(len(data["pitch"])) * 0.1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top Figure: Target vs Actual Pitch ---
    ax1.plot(time_seconds, data["target"], 'k--', label="Target Pitch", alpha=0.7)
    ax1.plot(time_seconds, data["pitch"], 'b-', label="Actual Pitch")
    ax1.set_ylabel("Pitch (rad)")
    ax1.set_title(f"Tracking Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Bottom Figure: Power & Action ---
    ax2.set_xlabel("Time (s)")

    # Secondary Y-Axis (Action)
    ax2.set_ylabel("Applied Action (V)")
    ax2.plot(time_seconds, data["action"], color="tab:red", alpha=0.6, linestyle='-', label="Action")

    ax2.set_title("Control Action")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def evaluate_agent(
        model,
        plot: bool = False,
        num_episodes: int = 10,
        stop_time: float = 200.0,
        seed: int | None = 42,
        flatten_observation: bool = False,
) -> dict:
    traj = RandomTrajectory(max_duration=stop_time, seed=seed)
    env = AeroSimulationEnv(
        render_mode="rgb_array",
        target_tilt=traj,
        stop_time=traj.total_duration,
        norm_action=True,
        norm_observation=True,
        initial_tilt=lambda: traj.get_value(0),
    )
    if flatten_observation:
        env = FlattenObservation(env)
    deviations = []
    powers = []
    actions = []
    for episode in range(num_episodes):
        data = run_episode(model, env)
        deviation = np.mean(np.abs(np.array(data["pitch"]) - np.array(data["target"])))
        deviations.append(deviation)
        powers.append(np.mean(data["power"]))
        actions.append(np.mean(np.abs(data["action"])))
        if plot:
            plot_episode(data)

    return {
        "deviation": deviations,
        "power": powers,
        "action": actions,
    }
