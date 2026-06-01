import gymnasium as gym
import numpy as np


class PreferenceConditionedPowerPenaltyWrapper(gym.Wrapper):
    """
    Wraps the AeroBaseEnv to append the preference power penalty weight to the Dict state
    and dynamically update the environment's reward scalarization.
    """

    def __init__(self, env):
        super().__init__(env)
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["alpha"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(new_obs_space)
        self.current_power_penalty_weight = 0.25

    def reset(self, *, seed=None, options=None):
        if options is not None and "power_penalty_weight" in options:
            self.current_power_penalty_weight = options["power_penalty_weight"]
        else:
            self.current_power_penalty_weight = np.random.uniform(0.0, 1.0)

        self.env.unwrapped.power_penalty_weight = self.current_power_penalty_weight
        obs, info = self.env.reset(seed=seed, options=options)
        obs["alpha"] = np.array([self.current_power_penalty_weight], dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["alpha"] = np.array([self.current_power_penalty_weight], dtype=np.float32)
        return obs, reward, terminated, truncated, info
