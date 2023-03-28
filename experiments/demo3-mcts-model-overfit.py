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
# Load sequences
sequences = []
names = ['', '_second', '_third', '_four', '_five', '_six', '_seven', '_eight']
for n in names:
    fname = f'../data/mcts-w-crashes/mcts_dataset_alldata{n}.npy'
    A = np.load(fname, allow_pickle=True)
    sequence = [load_sequence(row) for row in A]
    sequences += sequence
print(len(sequences))

# load model
checkpoint = torch.load('saved_models/checkpoint-mlp-decision-transformer-expert-mcts.pth')

config = {
    'device': 'cpu',#'cuda',
    'eval_render': False,
    'mode': 'normal',
    'experiment_name': 'mlp-decision-transformer-expert-mcts',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 2000,
    'num_steps_per_iter': 50,#10000,
    'context_length': 20,
    'batch_size': 64,
    'num_eval_episodes': 10,
    'pct_traj': 1.0,
    'n_layer': 3,
    'embed_dim': 128,
    'n_head': 4,
    'activation_function': 'relu',
    'dropout': 0.5,
    'model': checkpoint['model'],
    'optimizer': checkpoint['optimizer'],
    'learning_rate': 1e-6,
    'warmup_steps': 100,#10000,
    'weight_decay': 1e-6,
    'env_targets': [0.8, 1.0, 1.3],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
    'log_highest_return': True,
}


env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.config["duration"] = 100
env.config['policy_frequency'] = 1

sample = True
(state, info), done = env.reset(), False
state_dim = 25
act_dim = 5


model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=config['context_length'],
        max_ep_len=60,
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
num_timesteps = sum(traj_lens)

target_return = 1.0
n_evals = 10

# run eval episodes
for i in range(n_evals):
    evaluate_episode_rtg_nonstop(
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
        sample=True,
    )
