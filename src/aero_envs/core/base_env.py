from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
from typing import SupportsFloat
from collections.abc import Callable
from collections import deque
from gymnasium.core import ObsType
from decimal import Decimal
from ..utils.trajectories import TrajectoryGenerator, RandomTrajectory


class AeroBaseEnv(gym.Env, ABC):
    """
    Base class for the Quanser Aero 2 environment.
    """

    MAX_OBSERVATION_VALUES = {
        "pitch": np.pi / 2,
        "velocity": 0.24408247,
        "target": np.pi / 2,
    }

    def __init__(
        self,
        target_tilt: int | float | TrajectoryGenerator = RandomTrajectory(),
        initial_tilt: int | float | Callable[[], float] = 0,
        sample_time: float = 0.1,
        stop_time: float = 60.0,
        norm_action: bool = False,
        norm_observation: bool = False,
        power_penalty_weight: float = 0.0,
    ):
        super().__init__()


        if isinstance(target_tilt, float) or isinstance(target_tilt, int):
            if np.abs(target_tilt) > np.pi / 4:
                raise ValueError("target_tilt must be within -pi/4 and pi/4 radians")
            self.target_tilt_fn = TrajectoryGenerator()
            self.target_tilt_fn.add_step(target_tilt, duration=stop_time)
        elif isinstance(target_tilt, TrajectoryGenerator):
            self.target_tilt_fn = target_tilt
        else:
            raise ValueError("target_tilt must be a int, a float or a TrajectoryGenerator")
        self.target_tilt = self.target_tilt_fn(0)

        if isinstance(initial_tilt, float) or isinstance(initial_tilt, int):
            if np.abs(initial_tilt) > np.pi / 4:
                raise ValueError("initial_tilt must be within -pi/4 and pi/4 radians")
            self.initial_tilt = initial_tilt
            self.initial_tilt_fn = None
        elif callable(initial_tilt):
            self.initial_tilt = initial_tilt()
            self.initial_tilt_fn = initial_tilt
        else:
            raise ValueError("initial_tilt must be a int, a float or a callable")

        max_action = 1.0 if norm_action else 24.0
        self.action_space = gym.spaces.Box(low=-max_action, high=max_action, shape=(1,))

        observation_space_ranges = {
            key: (-1, 1) if norm_observation else (-value, value)
            for key, value in self.MAX_OBSERVATION_VALUES.items()
        }

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                key: gym.spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
                for key, (low, high) in observation_space_ranges.items()
            }
        )

        self.stop_time = stop_time
        self.sample_time = sample_time
        self.current_time = 0.0
        self.norm_action = norm_action
        self.norm_observation = norm_observation
        self.power_penalty_weight = power_penalty_weight

        pitches_buffer_size = np.ceil(
            60 / sample_time if np.isinf(stop_time) else stop_time / sample_time
        )
        self.pitches = deque(maxlen=int(pitches_buffer_size))
        self.actions = deque(maxlen=int(1 // sample_time))

    @property
    def pitch(self) -> float:
        return self.pitches[-1] if self.pitches else self.initial_tilt

    @property
    def last_pitch(self) -> float:
        return self.pitches[-2] if len(self.pitches) > 1 else self.pitch

    @property
    def state(self) -> ObsType:
        state = {
            "pitch": np.array([self.pitch], dtype=np.float32),
            "velocity": np.array([self.last_pitch - self.pitch], dtype=np.float32),
            "target": np.array([self.target_tilt], dtype=np.float32),
        }

        if self.norm_observation:
            state = {
                key: value / self.MAX_OBSERVATION_VALUES[key]
                for key, value in state.items()
            }

        return state

    @property
    def reward(self) -> SupportsFloat:
        e = -np.abs(self.target_tilt - self.pitch)
        power_penalty = -self.power / (24 * 3.9197)
        r = (1 - self.power_penalty_weight) * e + self.power_penalty_weight * power_penalty
        if self.truncated:
            reward_horizon = min(self.stop_time - self.current_time, 1.0)
            return reward_horizon / self.sample_time * r
        return r

    @property
    def terminated(self) -> bool:
        return self.current_time >= self.stop_time

    @property
    def truncated(self) -> bool:
        return np.abs(self.pitch) >= np.pi / 2

    def increment_timer(self) -> None:
        """
        Increment the current time by the sample time and round it to prevent floating point errors.
        """
        self.current_time += self.sample_time
        # round the current time with the same precision as the sample time to prevent floating point errors
        self.current_time = np.round(
            self.current_time,
            -Decimal(str(self.sample_time)).normalize().as_tuple().exponent,
        )

    def update_target_tilt(self) -> None:
        """
        Update the target tilt based on the target tilt function.
        """
        if self.target_tilt_fn is not None:
            self.target_tilt = self.target_tilt_fn(self.current_time)

    @property
    @abstractmethod
    def power(self) -> float:
        pass
