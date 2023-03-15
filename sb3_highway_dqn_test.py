# import gym
import highway_env
import gymnasium as gym
# # Agent
from stable_baselines3 import DQN
from stable_baselines3 import PPO
highway_env.register_highway_envs()
import numpy as np
import csv

def main():
    # Create the environment
    env = gym.make("highway-v0")

    # model = DQN('MlpPolicy', env,
    #             policy_kwargs=dict(net_arch=[256, 256]),
    #             learning_rate=5e-4,
    #             buffer_size=15000,
    #             learning_starts=200,
    #             batch_size=32,
    #             gamma=0.8,
    #             train_freq=1,
    #             gradient_steps=1,
    #             target_update_interval=50,
    #             verbose=1)

    # model.learn(int(2e4))
    # model.save("highway_dqn/model")
    
        # Load and test saved model
    model = PPO.load("highway_dqn/ppo_model")
    # while True:
    #     done = truncated = False
    #     obs, info = env.reset()
    #     while not (done or truncated):
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, info = env.step(action)
    #         print(obs,reward, action, '\n')
    #         env.render()
    datatest = []
    for _ in range(15):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            datatest.append([obs, action, reward])
            print("obs:", obs, "action:", action, "reward:", reward, '\n')
            env.render()
          
    with open('ppo_dataset.csv', 'w', newline='') as csvfile:

        # Create a CSV writer object
        csvwriter = csv.writer(csvfile, delimiter=',')

        # Write the list to the CSV file
        csvwriter.writerow(datatest)
        
if __name__ == "__main__":
    main()
