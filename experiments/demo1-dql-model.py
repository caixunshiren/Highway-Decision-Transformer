import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
import numpy as np
# from utils import RecordVideo
import torch
from modules.decision_transformer import DecisionTransformer
import pickle
from pipelines.evaluation.evaluate_episodes import evaluate_episode_rtg, evaluate_episode_rtg_nonstop


# load training sequences
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
    rewards = np.array(rewards) if not crashed else -np.array(rewards)
    dones = np.zeros_like(rewards)
    dones[-1] = 1
    sequence = {'states': states, 'actions': actions, 'rewards': rewards, 'dones': dones}
    return sequence
# Load sequences
A = np.load('../data/dataset_5000.npy', allow_pickle=True)
sequences = [load_sequence(row) for row in A if row[-1] is False]

# load model
checkpoint = torch.load('saved_models/checkpoint-mlp-decision-transformer.pth', map_location='cpu', pickle_module=pickle)
config = {
    'device': 'cpu',#'cuda',
    'mode': 'normal',
    'experiment_name': 'mlp-decision-transformer',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 100,
    'num_steps_per_iter': 10000,#10000,
    'context_length': 30,
    'batch_size': 32,
    'num_eval_episodes': 50,
    'pct_traj': 1.0,
    'n_layer': 3,
    'embed_dim': 128,
    'n_head': 4,
    'activation_function': 'relu',
    'dropout': 0.1,
    'model': checkpoint['model'],
    'optimizer': None,
    'learning_rate': 1e-4,
    'warmup_steps': 10000,#10000,
    'weight_decay': 1e-4,
    'env_targets': [],#[0.5, 1.0, 5.0, 10],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
}

# env = gym.make("highway-fast-v0", render_mode="rgb_array")

env_kwargs = {
    'id': "highway-v0",
    'config': {
        "lanes_count": 3,
        "vehicles_count": 15,
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
        "duration": 30,
    }
}

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env



env = make_configure_env(**env_kwargs)

print(env.config)


(state, info), done = env.reset(), False
state_dim = 70
act_dim = 5


model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=config['context_length'],
        max_ep_len=80,
        action_tanh=config['action_tanh'],
        hidden_size=config['embed_dim'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_inner=4 * config['embed_dim'],
        activation_function=config['activation_function'],
        n_positions=1024,
        resid_pdrop=config['dropout'],
        attn_pdrop=config['dropout'],
    )

model.load_state_dict(config['model'])


# get some useful statistics from training set
max_ep_len = max([len(path['states']) for path in sequences])  # take it as the longest trajectory
scale = np.mean([len(path['states']) for path in sequences])  # scale for rtg

# save all sequence information into separate lists
states, traj_lens, returns = [], [], []
for path in sequences:
    if config['mode'] == 'delayed':  # delayed: all rewards moved to end of trajectory
        path['rewards'][-1] = path['rewards'].sum()
        path['rewards'][:-1] = 0.
    states.append(path['states'])
    traj_lens.append(len(path['states']))
    returns.append(path['rewards'].sum())
traj_lens, returns = np.array(traj_lens), np.array(returns)

# used for input normalization
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
print(state_mean.shape, state_std.shape)
num_timesteps = sum(traj_lens)


highway_env.register_highway_envs()

target_return = 0.9
n_evals = 10

ret = []

# run eval episodes
for i in range(n_evals):
    r,_,_ = evaluate_episode_rtg_nonstop(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=max_ep_len,
        state_mean=state_mean,
        state_std=state_std,
        device='cpu',
        target_return=target_return,
        render=True,
        sample=False,
    )
    ret.append(r)
print(np.mean(ret))