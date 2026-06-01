from aero_envs.core.base_env import AeroBaseEnv
import time
import numpy as np
from array import array
from typing import Any, SupportsFloat, Callable
from gymnasium.core import ActType, ObsType

from aero_envs.utils.trajectories import RandomTrajectory, TrajectoryGenerator


class AeroHilEnv(AeroBaseEnv):
    COUNTS_PER_REV = 2048

    def __init__(
            self,
            card_identifier: str = "0@tcpip://localhost:18950?connection_timeout=5&d0=digital&d1=digital&led=auto&update_rate=normal&decimation=1",
            target_tilt: int | float | TrajectoryGenerator = RandomTrajectory(),
            initial_tilt: int | float | Callable[[], float] = 0,
            sample_time: float = 0.1,
            stop_time: float = 60.0,
            norm_action: bool = True,
            norm_observation: bool = True,
            power_penalty_weight: float = 0.0,
            **kwargs,
    ):
        super().__init__(
            target_tilt=target_tilt,
            initial_tilt=initial_tilt,
            sample_time=sample_time,
            stop_time=stop_time,
            norm_action=norm_action,
            norm_observation=norm_observation,
            power_penalty_weight=power_penalty_weight,
        )
        try:
            from quanser.hardware import HIL, HILError
            self._HIL = HIL
            self._HILError = HILError
        except ImportError as e:
            raise ImportError("Quanser Hardware Interface Library is required for AeroHilEnv.") from e

        self.input_channels = array("i", [0, 1])
        self.num_input_channels = len(self.input_channels)

        self.timer = time.perf_counter()

        try:
            self.card = self._HIL(
                card_type="quanser_aero2_usb", card_identifier=card_identifier
            )
        except self._HILError as e:
            print("Unable to open board:", e.get_error_message())
            raise e

        self.activate_motors()

    def activate_motors(self) -> None:
        try:
            values = array("b", [1] * self.num_input_channels)
            self.card.set_digital_directions(
                None, 0, self.input_channels, self.num_input_channels
            )
            self.card.write_digital(
                self.input_channels, self.num_input_channels, values
            )
        except self._HILError as e:
            print("Unable to write channels:", e.get_error_message())
            raise e

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        a = action[0] * 24 if self.norm_action else action[0]
        self.send_action(a)

        target_time = self.timer + self.sample_time
        while time.perf_counter() < target_time:
            pass

        self.timer = time.perf_counter()
        self.increment_timer()

        self.pitches.append(self.get_pitch())
        self.update_target_tilt()

        return self.state, self.reward, self.terminated, self.truncated, {}

    def send_action(self, action: float) -> None:
        voltages = array("d", [0.0] * self.num_input_channels)
        voltages[0] = action
        voltages[1] = -action
        try:
            self.card.write_analog(
                self.input_channels, self.num_input_channels, voltages
            )
        except self._HILError as e:
            print("Unable to write action:", e.get_error_message())

    def get_pitch(self):
        try:
            buffer = array("i", [0])
            self.card.read_encoder(array("I", [2]), 1, buffer)
            return buffer[0] * (2 * np.pi / self.COUNTS_PER_REV)
        except self._HILError as e:
            print("Unable to read pitch:", e.get_error_message())
            return 0.0

    def get_motor_currents(self) -> float:
        try:
            buffer = array("d", [0.0])
            self.card.read_analog(array("I", [0]), 1, buffer)
            return buffer[0]
        except self._HILError as e:
            print("Unable to read currents:", e.get_error_message())
            return 0.0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.pitches.clear()
        self.send_action(0.0)

        initial_pitch = self.get_pitch()
        self.pitches.append(initial_pitch)

        self.timer = time.perf_counter()
        return self.state, {}

    def close(self):
        self.send_action(0)  # Ensure motors stop
        self.card.close()

    @property
    def power(self) -> float:
        current = self.get_motor_currents()
        return abs(24.0 * current)
