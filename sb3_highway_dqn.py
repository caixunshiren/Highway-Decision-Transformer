# import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Categorical
import highway_env
import gymnasium as gym
# # Agent
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3 import PPO
highway_env.register_highway_envs()


def main():
    # Create the environment
    # env = gym.make("highway-v0")
    
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=2)

    model.learn(int(2e4))
    model.save("highway_dqn/PPO_model")
    
        # Load and test saved model
    # model = DQN.load("highway_dqn/model")
    # while True:
    # done = truncated = False
    # obs, info = env.reset()
    # while not (done or truncated):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = env.step(action)
    #     env.render()

if __name__ == "__main__":
    main()
