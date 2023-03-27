import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import pprint


highway_env.register_highway_envs()
env = gym.make('highway-v0', render_mode='rgb_array')
env.config["lanes_count"] = 3
env.config['screen_height'] = 500
env.config['vehicles_count'] = 50
env.reset()

for _ in range(300):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()

def test(model, iters, num_cars, num_lanes):
    '''Tests model online with highway env
    '''
    highway_env.register_highway_envs()
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.reset()

    for i in range(iters):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

