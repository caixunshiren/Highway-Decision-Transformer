from rl_agents.agents.common.factory import agent_factory
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
import numpy as np
# from utils import RecordVideo

highway_env.register_highway_envs()


def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env


env_kwargs = {
    'id': 'highway-v0',
    'config': {
        "lanes_count": 5,
        "vehicles_count": 100,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": [
                "presence",
                "x",
                "y",
                "vx",
                "vy",
                "cos_h",
                "sin_h"
            ],
            "absolute": False
        },
        "policy_frequency": 2,
        "duration": 60,
    }
}

# Make environment
# env = gym.make("highway-fast-v0", render_mode="rgb_array")
# env.config["duration"] = 60
env = make_configure_env(**env_kwargs)


env = RecordVideo(env, video_folder="run",
                  episode_trigger=lambda e: True)  # record all episodes

# Provide the video recorder to the wrapped environment
# so it can send it intermediate simulation frames.
env.unwrapped.set_record_video_wrapper(env)

(obs, info), done = env.reset(), False

# Make agent
# agent_config = {
#     # "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
#     "env_preprocessors": [{"method": "simplify"}],
#     "budget": 50,
#     "gamma": 0.7,
# }


agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
    "env_preprocessors": [{"method": "simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

dataset = []
all_data = []
# Run episode
for episode in range(10):
    obs, info = env.reset()
    done = truncated = False
    episode_data = []
    total_reward = 0
    while not (done or truncated):
    # for step in range(env.unwrapped.config["duration"]):
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_data.append([obs, action, reward])
        total_reward += reward
        # if done or truncated:
        #     break
        env.render()
    print('Episode: ', episode, ', Crashed? ', info['crashed'], ', Episode reward: ', round(total_reward,3))
    all_data.append(episode_data)
    if not info['crashed']:
        dataset.append(episode_data)
    # np.save('mcts_dataset_expert_eight.npy', np.array(dataset, dtype=object), allow_pickle=True)
    # np.save('mcts_dataset_alldata_eight.npy', np.array(all_data, dtype=object), allow_pickle=True)


env.close()
np.save('mcts_dataset_expert_eight.npy', np.array(dataset, dtype=object), allow_pickle=True)
np.save('mcts_dataset_alldata_eight.npy', np.array(all_data, dtype=object), allow_pickle=True)



