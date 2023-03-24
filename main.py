import numpy as np

# b = np.load('dataset_5000.npy', allow_pickle=True)

# for row in b:
#     for state, action, reward in row:
#         print(state, action, reward)
        
        
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
import numpy as np
# from utils import RecordVideo

highway_env.register_highway_envs()


# Make environment
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.config["duration"] = 60

print(env.config)


(obs, info), done = env.reset(), False

print(obs.shape, info)  
