"""
Script for Experiment 1: MLP encoder Decision Transformer on categorical action space
"""

import numpy as np
import torch
from pipelines.train_dt import train

config = {
    'device': 'cpu',#'cuda',
    'mode': 'normal',
    'experiment_name': 'mlp-decision-transformer',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 10,
    'num_steps_per_iter': 1000,#10000,
    'context_length': 20,
    'batch_size': 32,
    'num_eval_episodes': 50,
    'pct_traj': 1.0,
    'n_layer': 3,
    'embed_dim': 128,
    'n_head': 1,
    'activation_function': 'relu',
    'dropout': 0.1,
    'model': None,
    'optimizer': None,
    'scheduler': None,
    'learning_rate': 1e-4,
    'warmup_steps': 1000,#10000,
    'weight_decay': 1e-4,
    'env_targets': [],#[0.5, 1.0, 5.0, 10],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
}

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
