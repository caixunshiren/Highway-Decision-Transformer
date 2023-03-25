"""
Script for Experiment 1: MLP encoder Decision Transformer on categorical action space
"""

import numpy as np
import torch
import gymnasium as gym
import highway_env

from pipelines.train_dt import train
from pipelines.loading_utils import load_sequence

# check if cuda is available
print(torch.version.cuda)
print('cuda availability:', torch.cuda.is_available())

checkpoint = torch.load('saved_models/checkpoint-mlp-decision-transformer-expert-mcts.pth', map_location=torch.device('cpu'))

# set up environment
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.config["duration"] = 59

config = {
    'device': 'cpu',#'cuda',
    'env': env,
    'eval_render': False,
    'mode': 'normal',
    'experiment_name': 'mlp-decision-transformer-expert-mcts-distilled-3',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 1000,
    'num_steps_per_iter': 50,#10000,
    'context_length': 10,#15,
    'batch_size': 32,#64,
    'num_eval_episodes': 10,
    'pct_traj': 1.0,
    'n_layer': 2,
    'embed_dim': 20,
    'n_head': 1,
    'activation_function': 'relu',
    'dropout': 0.2,#0.3,
    'model': checkpoint['model'],
    'optimizer': checkpoint['optimizer'],
    'learning_rate': 3e-6,#1e-5,
    'warmup_steps': 100,#10000,
    'weight_decay': 1e-4,
    'env_targets': [0.8, 1.0, 1.3],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
    'log_highest_return': True,
}

# Load sequences
sequences = []
names = ['', '_second', '_third', '_four', '_five', '_six', '_seven', '_eight']
for n in names:
    fname = f'../data/mcts-w-crashes/mcts_dataset_alldata{n}.npy'
    A = np.load(fname, allow_pickle=True)
    sequence = [load_sequence(row) for row in A]
    sequences += sequence
print(len(sequences))

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
