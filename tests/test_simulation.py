import pytest
import gymnasium as gym
import numpy as np
from aero_envs import AeroSimulationEnv


class TestAeroSimulationEnv:
    """Test suite for AeroSimulationEnv"""

    @pytest.fixture
    def default_env(self):
        """Fixture providing a default environment instance"""
        return AeroSimulationEnv(
            step_size=0.01,
            stop_time=10.0,
            sample_time=0.2,
            target_tilt=0.0,
            initial_tilt=0.2,
            render_mode=None,
        )

    @pytest.fixture
    def human_render_env(self):
        """Fixture for environment with human render mode"""
        return AeroSimulationEnv(render_mode="human")

    def test_env_initialization(self, default_env):
        """Test environment initialization with various parameters"""
        env = default_env

        # Test basic inheritance
        assert isinstance(env, gym.Env)

        # Test parameter assignment
        assert env.step_size == 0.01
        assert env.stop_time == 10.0
        assert env.sample_time == 0.2
        assert env.target_tilt == 0.0
        assert env.input.phi0 == 0.2
        assert env.current_time == 0.0

    def test_env_reset(self, default_env):
        """Test environment reset functionality"""
        env = default_env
        obs, info = env.reset()

        # Test observation structure
        assert isinstance(obs, dict)
        expected_keys = {"pitch", "velocity", "target"}
        assert set(obs.keys()) == expected_keys

        # Test info structure
        assert isinstance(info, dict)

        # Test environment state after reset
        assert env.current_time == 0.0
        assert env.input.v0 == 0.0
        assert env.input.v1 == 0.0
        assert env.input.phi0 == 0.2

    def test_env_step(self, default_env):
        """Test environment step functionality"""
        env = default_env
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Test observation structure and types
        assert isinstance(obs, dict)
        expected_keys = {"pitch", "velocity", "target"}
        assert set(obs.keys()) == expected_keys

        # Test return values - allow numpy booleans
        assert isinstance(reward, (float, int))
        assert isinstance(terminated, (bool, np.bool_))  # Allow both bool types
        assert isinstance(truncated, (bool, np.bool_))  # Allow both bool types
        assert isinstance(info, dict)

        # Test environment state after step
        assert not bool(terminated), "Environment should not terminate after first step"
        assert not bool(truncated), "Environment should not truncate after first step"
        assert env.input.v0 == action[0]
        assert env.input.v1 == -action[0]
        assert env.current_time == env.sample_time

    def test_env_render(self, human_render_env):
        """Test environment rendering (should not raise exceptions)"""
        env = human_render_env
        env.reset()

        # Test that render doesn't crash
        try:
            env.render()
        except Exception as e:
            pytest.fail(f"Rendering failed with exception: {e}")

    def test_action_space(self, default_env):
        """Test action space properties"""
        env = default_env

        assert env.action_space.shape == (1,)
        assert isinstance(env.action_space, gym.spaces.Box)

        # Test valid actions
        valid_action = env.action_space.sample()
        assert env.action_space.contains(valid_action)
        assert env.action_space.contains([24.0])

        # Test invalid actions (outside bounds)
        assert not env.action_space.contains([24.1])
        assert not env.action_space.contains([-24.1])

    def test_observation_space(self, default_env):
        """Test observation space properties"""
        env = default_env

        # Test observation space type and shape
        assert isinstance(env.observation_space, gym.spaces.Dict)

        # Test that reset observation is in observation space
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

        # Test that step observation is in observation space
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)

        # Test invalid observation
        invalid_obs = [4, 5, 6]  # Should be dict, not list
        assert not env.observation_space.contains(invalid_obs)

    def test_observation_structure(self, default_env):
        """Test detailed observation structure"""
        env = default_env
        obs, _ = env.reset()

        # Test each observation component
        for key in ["pitch", "velocity", "target"]:
            assert key in obs
            value = obs[key]
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.float32
            assert value.shape == (1,)

    @pytest.mark.parametrize("norm_observation", [True, False])
    def test_observation_normalization(self, norm_observation):
        """Test observation normalization"""
        env = AeroSimulationEnv(norm_observation=norm_observation)
        obs, _ = env.reset()

        if norm_observation:
            # Check normalized values are in reasonable range
            for key, value in obs.items():
                assert np.all(
                    np.abs(value) <= 1.0
                ), f"Normalized {key} out of expected range"

        env.close()


class TestAeroSimulationEnvIntegration:
    """Integration tests for AeroSimulationEnv with gymnasium"""

    def test_gym_registration(self):
        """Test that environment is properly registered with gymnasium"""
        # This should not raise an exception
        env = gym.make("AeroSimulationEnv-v0", render_mode=None)

        # Gymnasium wraps environments with OrderEnforcing and PassiveEnvChecker
        # Check the unwrapped environment is our custom type
        assert hasattr(env, "unwrapped"), "Environment should have unwrapped attribute"
        assert isinstance(
            env.unwrapped, AeroSimulationEnv
        ), "Unwrapped environment should be AeroSimulationEnv"

        env.close()

    def test_full_episode(self):
        """Test running a complete episode"""
        env = gym.make("AeroSimulationEnv-v0", render_mode=None)
        obs, info = env.reset()

        done = False
        step_count = 0
        max_steps = 100  # Limit for test

        while not done and step_count < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)  # Convert to bool for condition
            step_count += 1

            # Test that observation remains valid throughout episode
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (float, int))

        env.close()
