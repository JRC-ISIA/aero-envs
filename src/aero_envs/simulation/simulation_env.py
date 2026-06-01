from ..core.base_env import AeroBaseEnv
from ..core.aero_interface import get_model_path, AeroInput, AeroOutput
import gymnasium as gym
import numpy as np
from typing import SupportsFloat, Any
from collections.abc import Callable
from gymnasium.core import ObsType
import ctypes as ct
import pygame
import math
from pathlib import Path
from ..utils.trajectories import RandomStepTrajectory, TrajectoryGenerator, RandomTrajectory, RandomTrajectoryBase


class AeroSimulationEnv(AeroBaseEnv):
    """
    Custom Gym environment for the Simulation of the Quanser Aero 2.

    :param step_size: Fixed time step size in seconds for the simulation, defaulting to 0.02 seconds.
    :param stop_time: Time duration in seconds for the simulation, defaulting to 15.0 seconds.
    :param sample_time: Sample time in seconds determining the frequency of interactions with the agent.
    :param target_tilt: Target angle in radians or a callable returning the target angle, defaulting to -10 degrees.
    :param initial_tilt: Initial angle in radians or a callable returning the initial angle, defaulting to 0 degrees.
    :param render_mode: Rendering mode, either "human", "rgb_array" or None, defaulting to None.
    :param norm_action: Whether to normalize the action space to [-1, 1].
    :param norm_observation: Whether to normalize the observation space to [-1, 1].
    :param power_penalty_weight: The weight of the power penalty
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        step_size: float = 0.02,
        stop_time: float = 15.0,
        sample_time: float = 0.1,
        target_tilt: int | float | TrajectoryGenerator = RandomStepTrajectory(),
        initial_tilt: int | float | Callable[[], float] = 0,
        render_mode: str | None = None,
        norm_action: bool = False,
        norm_observation: bool = False,
        power_penalty_weight: float = 0.0,
    ):
        super().__init__(
            target_tilt=target_tilt,
            initial_tilt=initial_tilt,
            sample_time=sample_time,
            stop_time=stop_time,
            norm_action=norm_action,
            power_penalty_weight=power_penalty_weight,
            norm_observation=norm_observation,
        )

        if not float(self.sample_time / step_size).is_integer():
            raise ValueError("sample_time must be a multiple of step_size")

        self.aero_dll = ct.CDLL(get_model_path())
        self.step_size = step_size

        self.render_mode = render_mode
        self.input = AeroInput.in_dll(self.aero_dll, "aero_U")
        self.input.phi0 = self.initial_tilt
        self.output = AeroOutput.in_dll(self.aero_dll, "aero_Y")

        # Initialize the pygame module
        if self.render_mode is not None:
            pygame.init()
            self.screen_width = 600
            self.screen_height = 400
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            if self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.internal_steps = int(self.sample_time / step_size)

        self.aero_dll.aero_initialize()
        self.aero_dll.aero_step()  # taking a first step to initialize the output

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        """
        Perform a step in the environment.
        :param action: The action to be performed in the environment.
        :return: The new state, the reward, whether the episode is terminated, whether the episode is truncated, and
            debug information.
        """

        voltage_multiplier = 24 if self.norm_action else 1
        voltage = np.clip(action[0] * voltage_multiplier, -24, 24)
        self.input.v0 = voltage
        self.input.v1 = -voltage

        self.update()
        self.pitches.append(self.output.pitch)
        self.actions.append(voltage)

        if self.render_mode == "human":
            self.render()

        return self.state, self.reward, self.terminated, self.truncated, {}

    def update(self):
        """
        Update the environment state.
        """
        self.update_target_tilt()

        for _ in range(self.internal_steps):
            self.aero_dll.aero_step()

        self.increment_timer()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict]:
        """
        Reset the environment to its initial state.
        :param seed: Seed for the random number generator.
        :param options:
        :return: The initial state of the environment and debug information.
        """
        super().reset(seed=seed)
        self.aero_dll.aero_terminate()
        self.aero_dll.aero_initialize()
        if self.initial_tilt_fn is not None:
            self.initial_tilt = self.initial_tilt_fn()
            self.input.phi0 = self.initial_tilt

        if isinstance(self.target_tilt_fn, RandomTrajectoryBase):
            self.target_tilt_fn.regenerate()

        self.input.v0 = 0.0
        self.input.v1 = 0.0
        self.output.velocity = 0.0
        self.output.pitch = self.initial_tilt
        self.pitches.clear()
        self.current_time = 0
        return self.state, {}

    def render(self) -> None | np.ndarray:
        """
        Render the environment. Requires the render_mode to be set to "human" or "rgb_array".
        :return: The rendered environment as a numpy array if render_mode is set to "rgb_array".
        """
        if self.render_mode is None:
            gym.logger.warn(
                "Render mode is not set, call the constructor with render_mode set to 'human' or 'rbg_array' to enable rendering."
            )
            return

        self.screen.fill((255, 255, 255))  # Fill the screen with white

        # Define the length and width of the lines and rectangles
        line_length = 200

        # Calculate the endpoints of the lines based on the output pitch
        x_center = self.screen_width // 2
        y_center = self.screen_height // 2

        # Calculate the endpoints of the target line based on the target pitch
        target_x_center = self.screen_width // 2
        target_y_center = self.screen_height // 2
        target_x_right = target_x_center + int(line_length * math.cos(self.target_tilt))
        target_y_right = target_y_center + int(line_length * math.sin(self.target_tilt))
        target_x_left = target_x_center - int(line_length * math.cos(self.target_tilt))
        target_y_left = target_y_center - int(line_length * math.sin(self.target_tilt))

        # Draw the beam
        package_root = Path(__file__).parent.parent
        image_path = package_root / "resources" / "images"
        base = pygame.image.load(image_path / "base.png")
        scaled_base = pygame.transform.scale(
            base, (base.get_width() // 3, base.get_height() // 3)
        )
        self.screen.blit(
            scaled_base,
            (
                x_center - scaled_base.get_width() // 2,
                self.screen_height - scaled_base.get_height(),
            ),
        )

        beam = pygame.image.load(image_path / "beam.png")
        scaled_beam = pygame.transform.scale(
            beam, (beam.get_width() // 3, beam.get_height() // 3)
        )
        beam_rect = scaled_beam.get_rect()
        beam_rect.x = x_center - scaled_beam.get_width() // 2
        beam_rect.y = y_center - 32

        rotated_beam = pygame.transform.rotate(
            scaled_beam, -self.output.pitch * 180 / math.pi
        )
        rotated_beam_rect = rotated_beam.get_rect(center=beam_rect.center)

        self.screen.blit(rotated_beam, rotated_beam_rect)

        # Draw the line for the target tilt
        pygame.draw.aaline(
            self.screen,
            (255, 0, 0),
            (target_x_right, target_y_right),
            (target_x_left, target_y_left),
        )

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.array3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        self.aero_dll.aero_terminate()
        if self.render_mode is not None:
            pygame.quit()

    @property
    def power(self) -> float:
        return self.input.v0 * self.output.i0 + self.input.v1 * self.output.i1
