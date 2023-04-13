import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import RecordVideo

import highway_env


def train_env():
    env = gym.make('highway-fast-v0', render_mode='rgb_array')
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 1,
        "duration": 60
    })
    env.reset()
    return env


def test_env():
    env = train_env()
    env.configure({"policy_frequency": 1, "duration": 60})
    env.reset()
    return env


if __name__ == '__main__':
    # Train
    train = False
    n_cpu = 6
    
    if train:
        
        env = make_vec_env(train_env, n_envs=n_cpu, seed=0,
                            vec_env_cls=SubprocVecEnv)
        model = DQN('CnnPolicy', env,
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    exploration_fraction=0.7,
                    verbose=1)
        model.learn(total_timesteps=int(1e7))
        model.save("highway_cnn/model")

    # Record video
    model = DQN.load("highway_cnn/model")


    env = test_env()
    env = RecordVideo(env, video_folder="highway_cnn/videos",
                      episode_trigger=lambda e: True, name_prefix="dqn-agent")  # record all episodes

    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    env.configure({"simulation_frequency": 15})
    env.unwrapped.set_record_video_wrapper(env)


    for i in range(10):
        # print(env.reset().shape)
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            # print(action)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    
        
    env.close()
