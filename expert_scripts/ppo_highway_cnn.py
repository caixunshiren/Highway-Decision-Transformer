import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env


def train_env():
    env = gym.make('highway-fast-v0')
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    })
    env.reset()
    return env


def test_env():
    env = train_env()
    env.configure({"policy_frequency": 15, "duration": 20})
    env.reset()
    return env


if __name__ == '__main__':
    # Train

    model = PPO('CnnPolicy', DummyVecEnv([train_env]),
                buffer_size=15000,
                batch_size=64,
                learning_rate=2e-3,
                verbose=2, device="cuda")
    
    # Train the agent
    model.learn(total_timesteps=int(2e6))
    model.save("highway_ppo_cnn/model")

    # Record video
    model = PPO.load("highway_ppo_cnn/model")

    env = DummyVecEnv([test_env])
    video_length = 2 * env.envs[0].config["duration"]
    env = VecVideoRecorder(env, "highway_ppo_cnn/videos/",
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix="ppo-agent")
    obs, info = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    env.close()
