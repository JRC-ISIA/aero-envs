from gymnasium.envs.registration import register

register(
    id="AeroSimulationEnv-v0",
    entry_point="aero_envs.simulation.simulation_env:AeroSimulationEnv",
)

register(
    id="AeroHilEnv-v0",
    entry_point="aero_envs.hil.hil_env:AeroHilEnv",
)

from .simulation.simulation_env import AeroSimulationEnv
from .hil.hil_env import AeroHilEnv

__version__ = "0.1.0"
