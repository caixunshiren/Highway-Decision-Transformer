import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
import numpy as np
# from utils import RecordVideo
import torch
from modules.decision_transformer import DecisionTransformer
import pickle
from pipelines.evaluation.evaluate_episodes import evaluate_episode_rtg, evaluate_episode_rtg_nonstop
from pipelines.loading_utils import load_sequence, get_action_count


# Load sequences
sequences = []
names = ['', '_second', '_third', '_four', '_five', '_six', '_seven', '_eight', '_nine']
for n in names:
    fname = f'../data/mcts-wo-crashes/mcts_dataset_expert{n}.npy'
    A = np.load(fname, allow_pickle=True)
    sequence = [load_sequence(row) for row in A]
    sequences += sequence
print(len(sequences))

# load model
checkpoint = torch.load('saved_models/checkpoint-mlp-decision-transformer-expert-mcts-distilled-5.pth', map_location='cpu', pickle_module=pickle)
config = {
    'device': 'cpu',#'cuda',
    'eval_render': True,
    'mode': 'normal',
    'experiment_name': 'mlp-decision-transformer-expert-mcts',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 15,
    'num_steps_per_iter': 100,#10000,
    'context_length': 20,
    'batch_size': 64,
    'num_eval_episodes': 5,
    'pct_traj': 1.0,
    'n_layer': 4,
    'embed_dim': 32,
    'n_head': 4,
    'activation_function': 'relu',
    'dropout': 0.2,
    'model': checkpoint['model'],
    'optimizer': checkpoint['optimizer'],
    'learning_rate': 1e-5,
    'warmup_steps': 100,#10000,
    'weight_decay': 1e-5,
    'env_targets': [0.8, 1.0, 1.2, 2.0],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
}

env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.config["duration"] = 60
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

target_return = 0.85
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
