import numpy as np
import gymnasium as gym
import highway_env


def load_sequence(row):
    '''
    Load a sequence from a row of the dataset.
    :param row: [[state1, action1, reward1], [state2, action2, reward2], ..., [stateN, actionN, rewardN]]
    :return: sequence: dict containing {'states': np.array([state1, state2, ..., stateT]),
                                       'actions': np.array([action1, action2, ..., actionT]),
                                       'rewards': np.array([reward1, reward2, ..., rewardT]),
                                       'dones': np.array([0,0, ..., 1])} -> trivial for our case as we always have one
                                       scene for each episode. Dones is also not used in experiments.
                    states: np.array of shape (T, *state_dim)
                    actions: np.array of shape (T, *action_dim)
                    rewards: np.array of shape (T, )
                    dones: np.array of shape (T, )
    '''
    states = []
    actions = []
    rewards = []
    for state, action, reward in row:
        # flatten state for mlp encoder
        states.append(state.reshape(-1))
        one_hot_action = np.zeros(5)
        one_hot_action[action] = 1
        actions.append(one_hot_action)
        rewards.append(reward)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.zeros_like(rewards)
    dones[-1] = 1
    sequence = {'states': states, 'actions': actions, 'rewards': rewards, 'dones': dones}
    return sequence


def load_sequence_w_crashes(row):
    '''
    Load a sequence from a row of the dataset.
    :param row: [[state1, action1, reward1], [state2, action2, reward2], ..., [stateN, actionN, rewardN]]
    :return: sequence: dict containing {'states': np.array([state1, state2, ..., stateT]),
                                       'actions': np.array([action1, action2, ..., actionT]),
                                       'rewards': np.array([reward1, reward2, ..., rewardT]),
                                       'dones': np.array([0,0, ..., 1])} -> trivial for our case as we always have one
                                       scene for each episode. Dones is also not used in experiments.
                    states: np.array of shape (T, *state_dim)
                    actions: np.array of shape (T, *action_dim)
                    rewards: np.array of shape (T, )
                    dones: np.array of shape (T, )
    '''
    crashed = row[-1]
    states = []
    actions = []
    rewards = []
    for state, action, reward in row[:-1]:
        # flatten state for mlp encoder
        states.append(state.reshape(-1))
        one_hot_action = np.zeros(5)
        one_hot_action[action] = 1
        actions.append(one_hot_action)
        rewards.append(reward)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.zeros_like(rewards)
    dones[-1] = 1
    sequence = {'states': states, 'actions': actions, 'rewards': rewards, 'dones': dones}
    return sequence


def get_action_count(sequences):
    actions = np.zeros(5)
    for sequence in sequences:
        actions += np.sum(sequence['actions'], axis=0)
    return actions


def grayscale_train_env():
    env = gym.make('highway-v0')
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 70, # 60

        # Hard Scenario Using Settings below
        # "vehicles_density": 1.2,
        # "vehicles_count": 25,
        "high_speed_reward": 0.5,
        "collision_reward": -2.5,
        "lane_change_reward": -0.05,
    })
    env.reset()
    return env
