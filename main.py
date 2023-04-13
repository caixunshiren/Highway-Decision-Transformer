from matplotlib import pyplot as plt
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
import numpy as np
from pipelines.loading_utils import load_sequence, get_action_count
# from utils import RecordVideo
import pprint


highway_env.register_highway_envs()

env_kwargs = {
    'id': 'highway-v0',
    'config': {
        "lanes_count": 3,
        "vehicles_count": 10,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": [
                "presence",
                "x",
                "y",
                "vx",
                "vy",
            ],
            "flatten": True,
            "absolute": False,
            "order": "sorted"
        },
        "vehicles_density": 2,
        "order": "order",

        "policy_frequency": 1,  # Hz
        "duration": 60,  # seconds
        "screen_width": 1200,  # [px]
        "screen_height": 300,  # [px]
        "show_trajectories": True,
    }
}


def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env


# Make environment
# env = gym.make("highway-fast-v0", render_mode="rgb_array")
# env.config["duration"] = 60
env = make_configure_env(**env_kwargs)

pprint.pprint(env.observation_space)


(obs, info), done = env.reset(), False

# sequence = [load_sequence(row) for row in obs]
# print(sequence)

pprint.pprint(obs)

plt.imshow(env.render())
plt.show()
print(env.action_space)

# for i in range(100):
#     action = env.action_space.sample()
#     # print(action)
#     pprint.pprint(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()
#     if done:
#         break
