from aero_envs.utils.evaluation import evaluate_agent
from stable_baselines3 import PPO
import numpy as np

if __name__ == "__main__":
    model_name = "ppo_example"
    model = PPO.load(f"example_models/{model_name}")
    results = evaluate_agent(
        model,
        plot=True,
        num_episodes=10,
        stop_time=200.0,
        seed=42,
    )
    print(f"Evaluation of {model_name}")
    print(f"Deviation: {np.mean(results['deviation']):.4f} ± {np.std(results['deviation']):.4f} rad")
    print(f"Power Consumption: {np.mean(results['power']):.4f} ± {np.std(results['power']):.4f} W")
    print(f"Action: {np.mean(results['action']):.4f} ± {np.std(results['action']):.4f} V")
