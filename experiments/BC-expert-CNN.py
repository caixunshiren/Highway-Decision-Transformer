"""
Script for Experiment 1: MLP encoder Decision Transformer on categorical action space
"""
import sys
sys.path.append('/Users/jwoo/Documents/GitHub/Highway-Decision-Transformer')

import numpy as np
import torch

from pipelines.train_bc_CNN import train

# check if cuda is available
print(torch.version.cuda)
print('cuda availability:', torch.cuda.is_available())

# checkpoint = torch.load('saved_models/checkpoint-BC-expert-mcts.pth')

config = {
    'device': 'cpu',#'cuda',
    'mode': 'normal',
    'experiment_name': 'BC-expert-mcts',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 10,
    'num_steps_per_iter': 10000,
    'context_length': 10,
    'batch_size': 32,
    'num_eval_episodes': 50,
    'pct_traj': 1.0,
    'n_layer': 3,
    'in_channels': 4,   #4 channels per timestep
    'channels': 32,
    'embed_dim': 256,
    'channels': 3,
    'activation_function': 'relu',
    'dropout': 0.1,
    # 'model': checkpoint['model'],
    # 'optimizer': checkpoint['optimizer'],
    'learning_rate': 1e-4,
    'warmup_steps': 1000,#10000,
    'weight_decay': 1e-4,
    'env_targets': [],#[0.5, 1.0, 5.0, 10],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=2).flatten()),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
}


# Load sequences
A = np.load('./data/visual_dataset_expert.npz', allow_pickle=True)

sequences = {
    'states': A['expert_observations'],
    'actions': A['expert_actions'],
    'returns': A['expert_rewards']
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