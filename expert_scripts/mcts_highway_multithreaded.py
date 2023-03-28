from rl_agents.agents.common.factory import agent_factory
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
import numpy as np
from multiprocessing import Pool
# from utils import RecordVideo

highway_env.register_highway_envs()

def run_episode(dataset):
    # Make environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.config["duration"] = 60

    # env = RecordVideo(env, video_folder="run",
    #                   episode_trigger=lambda e: True)  # record all episodes

    # # Provide the video recorder to the wrapped environment
    # # so it can send it intermediate simulation frames.
    # env.unwrapped.set_record_video_wrapper(env)

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
        "__class__": "",
        "env_preprocessors": [{"method": "simplify"}],
        "budget": 50,
        "gamma": 0.7,
    }
    agent = agent_factory(env, agent_config)

    dataset = []
    # Run episode
    for episode in range(2000):
        obs, info = env.reset()
        done = truncated = False
        episode_data = []
        while not (done or truncated):
        # for step in range(env.unwrapped.config["duration"]):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_data.append([obs, action, reward])
            # if done or truncated:
            #     break
            env.render()
        print('Episode: ', episode, ', Crashed?: ', info['crashed'])
        
        
        np.save(dataset, np.array(
            dataset, dtype=object), allow_pickle=True)

    env.close()
    np.save(dataset, np.array(
        dataset, dtype=object), allow_pickle=True)


run_episode()

