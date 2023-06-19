import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna

import highway_env

from tqdm.auto import tqdm



def train_env():
    env = gym.make('highway-v0')
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 70, # 60
        
        # Hard Scenario Using Settings below
        # "vehicles_density": 1.2,
        # "vehicles_count": 25,
        "high_speed_reward": 0.5,
        "collision_reward": -2.5,
        "lane_change_reward": -0.05,
    })
    env.reset()
    return env

def objective(trial):
    env = make_vec_env("highway-v0")

    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True)
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000)
    batch_size = trial.suggest_int("batch_size", 16, 256)
    gamma_size = trial.suggest_int("gamma", 0.)
    
    
                        learning_rate=5e-4,
                    buffer_size=30000,
                    learning_starts=2000,
                    batch_size=64,
                    gamma=0.85,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    exploration_fraction=0.6,
                    exploration_final_eps=0.01,
                    verbose=0,device = device, tensorboard_log=log_dir)

    # Instantiate DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        verbose=0,
    )

    # Train the model
    model.learn(total_timesteps=5000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return -mean_reward  # Optuna minimizes the objective function



if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as desired

    print("Best hyperparameters: ", study.best_params)
    print("Best mean reward: ", -study.best_value)
