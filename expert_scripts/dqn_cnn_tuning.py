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


import highway_env

from tqdm.auto import tqdm


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """

    def __init__(self, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self._plot = None

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if self._plot is None:  # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else:  # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
                                    self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True, True, True)
            self._plot[-1].canvas.draw()


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction


class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(
                        "Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True


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
    env.configure({"policy_frequency": 1, "duration": 60,
                  "show_trajectories": True, "simulation_frequency": 15})
    env.reset()
    return env


if __name__ == '__main__':
    # Train
    train = True
    n_cpu = 7
    
     # Create log dir
    log_dir = "highway_cnn/logs/"
    os.makedirs(log_dir, exist_ok=True)

    if train:

        env = make_vec_env(train_env, n_envs=n_cpu, seed=0,
                           vec_env_cls=SubprocVecEnv,  monitor_dir=log_dir)
        
        
        # model = DQN('CnnPolicy', env,
        #             learning_rate=5e-4,
        #             buffer_size=15000,
        #             learning_starts=200,
        #             batch_size=32,
        #             gamma=0.8,
        #             train_freq=1,
        #             gradient_steps=1,
        #             target_update_interval=50,
        #             exploration_fraction=0.7,
        #             verbose=1)
        
        model = DQN('CnnPolicy', env,
                    learning_rate=1e-4,
                    buffer_size=50000,
                    learning_starts=1000,
                    target_update_interval=500,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=4,
                    gradient_steps=1,
                    exploration_fraction=0.4,
                    exploration_final_eps= 0.01,
                    verbose=0)
        
        plotting_callback = PlottingCallback()
        
        auto_save_callback = SaveOnBestTrainingRewardCallback(
            check_freq=5000, log_dir=log_dir, verbose=1)
        
        with ProgressBarManager(1e6) as progress_callback:
            # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
            model.learn(total_timesteps=int(1e6), callback=[
                        progress_callback, auto_save_callback, plotting_callback])
        
        
        model.save("highway_cnn/model_1e7")

    # Record video
    model = DQN.load("highway_cnn/model_1e7")

    env = test_env()
    env = RecordVideo(env, video_folder="highway_cnn/videos",
                      episode_trigger=lambda e: True, name_prefix="dqn-agent")  # record all episodes

    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    env.unwrapped.set_record_video_wrapper(env)

    for episode in range(10):
        # print(env.reset().shape)
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            # print(action)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

    env.close()
