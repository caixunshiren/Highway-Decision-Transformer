import numpy as np
import random
import torch
import wandb

from pipelines.evaluation.evaluate_episodes import evaluate_episode_rtg
from pipelines.training.bc_seq_trainer import SequenceTrainer
from modules.behaviour_cloning_CNN import BehaviourCloning

def discount_cumsum(x, gamma):
    """
    Compute discounted cumulative sums of future reward.
    adopted from https://github.com/kzl/decision-transformer/blob/master/gym/experiment.py
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def train(config, sequences, continue_training=False):
    """
    Train a decision transformer on a dataset.
    :param config: training configuration
    :param sequences: list containing [sequence1, sequence2, ..., sequenceN]
            sequence: dict containing {'states': np.array([state1, state2, ..., stateT]),
                                       'actions': np.array([action1, action2, ..., actionT]),
                                       'rewards': np.array([reward1, reward2, ..., rewardT])}
                    states: np.array of shape (T, *state_dim)
                    actions: np.array of shape (T, *action_dim)
                    rewards: np.array of shape (T, )
                    'dones': np.array([0,0, ..., 1])} -> trivial for our case as we always have one
                                       scene for each episode. Dones is also not used in experiments.

    code partially adapted from https://github.com/kzl/decision-transformer/blob/master/gym/experiment.py
    """
    assert sequences is not None, 'No sequences provided for training.'
    device = config['device']
    input_dim = config['channels'] * sequences['states'].shape[-1] * sequences['states'].shape[-2]
    states = sequences['states']
    returns = sequences['returns']
    actions = sequences['actions']

    # used for input normalization
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = len(sequences['states'])

    print('=' * 50)
    print(f'Starting new experiment: {config["experiment_name"]}')
    print(f'{num_timesteps} timesteps found')
    print('=' * 50)

    K = config['context_length']
    batch_size = config['batch_size']
    num_eval_episodes = config['num_eval_episodes']
    pct_traj = config['pct_traj']

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_timesteps),
            size=batch_size,
            replace=True
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            # get sequences from dataset
            si = batch_inds[i]
            s.append(states[si:si + max_len].reshape(-1, 128, 64))

            # padding and state + reward normalization
            s[-1] = np.concatenate([np.zeros((max_len * 4 - len(s[-1]), 128, 64)), s[-1]])
            s[-1] = (s[-1] - state_mean) / state_std

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret = length = 0
                    # ret, length = evaluate_episode_rtg(
                    #     state_dim,
                    #     act_dim,
                    #     model,
                    #     max_ep_len=max_ep_len,
                    #     scale=scale,
                    #     target_return=target_rew / scale,
                    #     mode=config['mode'],
                    #     state_mean=state_mean,
                    #     state_std=state_std,
                    #     device=device,)
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    print("input_dim:", input_dim, " K:", K)

    act_dim = 5
    n_layer = 3

    model = BehaviourCloning(
        input_dim = input_dim, 
        act_dim = act_dim, 
        hidden_size = config['embed_dim'], 
        n_layer = config['n_layer'], 
        dropout = config['dropout'], 
        max_length = K,
        in_channels = config['in_channels'] * K,    #4 channels per timestep x K timesteps
        channels = config['channels'],
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    if continue_training:
        model.load_state_dict(config['model'])
        optimizer.load_state_dict(config['optimizer'])
    else:
        model = model.to(device=device)
    warmup_steps = config['warmup_steps']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=config['loss_fn'],
        eval_fns=[eval_episodes(tar) for tar in config['env_targets']],
        err_fn=config['err_fn']
    )

    if config['log_to_wandb']:
        wandb.init(
            name=config["experiment_name"],
            group=config["group_name"],
            project='decision-transformer',
            config=config
        )
        # wandb.watch(model)  # wandb has some bug

    for iteration in range(config['max_iters']):
        outputs = trainer.train_iteration(num_steps=config['num_steps_per_iter'], iter_num=iteration+1, print_logs=True)
        if config['log_to_wandb']:
            wandb.log(outputs)

    return model, optimizer, scheduler
