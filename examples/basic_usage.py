import gymnasium as gym
import aero_envs  # noqa: F401

if __name__ == "__main__":
    env = gym.make("AeroSimulationEnv-v0", render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
