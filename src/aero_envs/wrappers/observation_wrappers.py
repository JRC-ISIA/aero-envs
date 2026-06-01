import numpy as np
import gymnasium as gym


class FutureTargetsWrapper(gym.ObservationWrapper):
    """
    Appends future target points to the observation dict.
    """

    def __init__(self, env: gym.Env, num_future_targets: int = 5, future_target_interval: float = 0.2):
        super().__init__(env)
        self.num_future_targets = num_future_targets
        self.future_target_interval = future_target_interval

        # Access the unwrapped base environment for configuration
        base_env = self.unwrapped

        if base_env.norm_observation:
            low, high = -1.0, 1.0
        else:
            limit = base_env.MAX_OBSERVATION_VALUES["target"]
            low, high = -limit, limit

        new_spaces = self.observation_space.spaces.copy()
        new_spaces["future_targets"] = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.num_future_targets,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, observation):
        # Create a copy so we don't mutate the underlying env's observation dict directly
        obs = observation.copy()
        base_env = self.unwrapped

        future_times = np.array([
            base_env.current_time + (i * self.future_target_interval)
            for i in range(1, self.num_future_targets + 1)
        ])

        future_vals = np.array(
            [base_env.target_tilt_fn(t) for t in future_times],
            dtype=np.float32
        )

        if base_env.norm_observation:
            future_vals = future_vals / base_env.MAX_OBSERVATION_VALUES["target"]

        obs["future_targets"] = future_vals
        return obs


class TargetVelocityWrapper(gym.ObservationWrapper):
    """
    Appends the target's current velocity to the observation dict.
    """

    def __init__(self, env: gym.Env, max_velocity: float = 1.0):
        super().__init__(env)
        self.max_velocity = max_velocity

        base_env = self.unwrapped

        if base_env.norm_observation:
            low, high = -1.0, 1.0
        else:
            low, high = -self.max_velocity, self.max_velocity

        new_spaces = self.observation_space.spaces.copy()
        new_spaces["target_velocity"] = gym.spaces.Box(
            low=low,
            high=high,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, observation):
        obs = observation.copy()
        base_env = self.unwrapped

        velocity = base_env.target_tilt_fn.get_derivative(base_env.current_time)

        if base_env.norm_observation:
            velocity = velocity / self.max_velocity

        obs["target_velocity"] = np.array([velocity], dtype=np.float32)
        return obs


class PowerPenaltyWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["power_penalty"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(new_obs_space)

    def observation(self, observation):
        obs = observation.copy()
        current_power_penalty = self.env.unwrapped.power_penalty_weight
        obs["power_penalty"] = np.array([current_power_penalty], dtype=np.float32)
        return obs
