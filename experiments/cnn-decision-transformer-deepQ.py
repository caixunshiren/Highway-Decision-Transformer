"""
Script for Experiment 1: MLP encoder Decision Transformer on categorical action space
"""

import numpy as np
import torch
import gymnasium as gym
import highway_env

from pipelines.train_dt import train
from pipelines.loading_utils import load_sequence, get_action_count, grayscale_train_env

# check if cuda is available
print(torch.version.cuda)
print('cuda availability:', torch.cuda.is_available())

checkpoint = torch.load('saved_models/best-25.71-checkpoint-mlp-decision-transformer-expert-mcts-distilled-8.pth', map_location=torch.device('cpu'))

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
        # if end_i - start_i < 130:
        #     # drop too short sequences
        #     continue
        s = obs[start_i:end_i]
        a = np.eye(5)[actions[start_i:end_i].astype(int)]
        r = rewards[start_i:end_i]
        d = dones[start_i:end_i]
        if np.sum(r) < 135:
            continue
        sequence = {'states': s, 'actions': a, 'rewards': r, 'dones': d}
        sequences.append(sequence)
    return sequences

sequences = make_sequence(expert_observations, expert_actions, expert_rewards, expert_done)

print(len(sequences))
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
    'num_steps_per_iter': 500,#10000,
    'context_length': 16,#15,
    'batch_size': 16,#64,
    'num_eval_episodes': 4,
    'pct_traj': 1.0,
    'n_layer': 4,
    'embed_dim': 64,
    'n_head': 4,
    'state_encoder': 'cnn',
    'activation_function': 'relu',
    'dropout': 0.2,#0.3,
    'model': checkpoint['model'],
    'optimizer': checkpoint['optimizer'],
    'learning_rate': 6e-5,
    'warmup_steps': 100,#10000,
    'weight_decay': 1e-5,
    'env_targets': [0.85, 1.0],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
    'log_highest_return': True,
    'input_type': 'grayscale',
    'eval_size': 10,
    'in_shape': in_shape,
}

# Train model
model, optimizer, scheduler = train(config, sequences, continue_training=False)

# save model as checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    }
torch.save(checkpoint, f'saved_models/checkpoint-{config["experiment_name"]}.pth')
print('-'*20+'model saved'+'-'*20)


##TODO: Noted issues:
# 1. predict_action should be linear or sigmoid not tanh -> solved by setting action_tanh=False
# 2. loss function should be cross entropy not mse on action -> solved by setting loss_fn to nn.CrossEntropyLoss()
# 3. save model, optimizer, and scheduler
