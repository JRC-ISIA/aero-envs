import gymnasium as gym
import aero_envs  # noqa: F401
from aero_envs.utils.trajectories import TrajectoryGenerator
from aero_envs.wrappers.observation_wrappers import FutureTargetsWrapper
import numpy as np

def build_trajectory() -> TrajectoryGenerator:
    traj = TrajectoryGenerator()
    traj.add_ramp(-20 * np.pi / 180, 10.0)
    traj.add_sine(amplitude=15 * np.pi / 180, frequency=0.05, duration=10.0)
    traj.add_ramp(0, 5.0)
    traj.add_step(35 * np.pi / 180, 10.0)
    traj.add_s_curve(0, 10.0)
    traj.add_hold(5.0)
    return traj

if __name__ == "__main__":

    trajectory = build_trajectory()
    trajectory.plot()

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
        action = [0.0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
