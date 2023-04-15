import numpy as np
import torch
import gymnasium as gym
import highway_env
from modules.decision_transformer import DecisionTransformer

from pipelines.evaluation.evaluate_episodes import evaluate_episode_rtg, evaluate_episode_rtg_nonstop
from pipelines.loading_utils import load_sequence, get_action_count, grayscale_train_env

# check if cuda is available
print(torch.version.cuda)
print('cuda availability:', torch.cuda.is_available())

checkpoint = torch.load('saved_models/best-41.46-checkpoint-cnn-decision-transformer-deepQ.pth', map_location=torch.device('cpu'))

# set up environment
env = grayscale_train_env()

# Load sequences
sequences = []

fname = '../data/deep-q-grayscale/expert_data_with_reward_done.npz'
A = np.load(fname, allow_pickle=True)
expert_actions = A['expert_actions']
expert_observations = A['expert_observations']
expert_rewards = A['expert_rewards']
expert_done = A['expert_done']
in_shape = expert_observations.shape[1:]

def make_sequence(obs, actions, rewards, dones):
    # flatten obs
    in_shape = obs.shape[1:]
    obs = obs.reshape(-1, np.prod(in_shape))
    sequences = []
    end_indices = np.where(dones)[0].tolist()
    if end_indices[-1] != len(actions):
        end_indices.append(len(actions))
    for start_i, end_i in zip([0] + end_indices[:-1], end_indices):
        s = obs[start_i:end_i]
        a = np.eye(5)[actions[start_i:end_i].astype(int)]
        r = rewards[start_i:end_i]
        d = dones[start_i:end_i]
        sequence = {'states': s, 'actions': a, 'rewards': r, 'dones': d}
        sequences.append(sequence)
    return sequences

sequences = make_sequence(expert_observations, expert_actions, expert_rewards, expert_done)

print(len(sequences))
sequences = sequences[:20]
# sequence statistics
action_counts = get_action_count(sequences)
print("action frequencies:", action_counts)

config = {
    'device': 'cpu',#'cuda',
    'env': env,
    'eval_render': True,
    'mode': 'normal',
    'experiment_name': 'cnn-decision-transformer-deepQ',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 1000,
    'num_steps_per_iter': 10,#10000,
    'context_length': 16,#15,
    'batch_size': 16,#64,
    'num_eval_episodes': 4,
    'pct_traj': 1,
    'n_layer': 4,
    'embed_dim': 64,
    'n_head': 4,
    'state_encoder': 'cnn',
    'activation_function': 'relu',
    'dropout': 0.2,#0.3,
    'model': checkpoint['model'],
    'optimizer': checkpoint['optimizer'],
    'learning_rate': 5e-5,
    'warmup_steps': 100,#10000,
    'weight_decay': 1e-5,
    'env_targets': [0.85, 1.0],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
    'log_highest_return': True,
    'input_type': 'grayscale',
    'eval_size': 3,
    'in_shape': in_shape,
}


sample = True
(state, info), done = env.reset(), False
state_dim = np.prod(in_shape)
act_dim = 5
K = config['context_length']
# get some useful statistics from training set
max_ep_len = max([len(path['states']) for path in sequences])  # take it as the longest trajectory
scale = np.mean([len(path['states']) for path in sequences])  # scale for rtg


model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        action_tanh=config['action_tanh'],
        hidden_size=config['embed_dim'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        state_encoder=config.get('state_encoder', None),
        in_shape=config.get('in_shape', None),
        n_inner=4 * config['embed_dim'],
        activation_function=config['activation_function'],
        n_positions=1024,
        resid_pdrop=config['dropout'],
        attn_pdrop=config['dropout'],
    )

model.load_state_dict(config['model'])


state_mean, state_std = np.array(50), np.array(100)

target_return = 0.85
n_evals = 10

# run eval episodes
for i in range(n_evals):
    evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        state_mean=state_mean,
        state_std=state_std,
        device='cpu',
        target_return=target_return,
        render=True,
        sample=False,
    )
