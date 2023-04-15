import numpy as np
import random
import torch
import wandb

from pipelines.evaluation.evaluate_episodes import evaluate_episode_rtg
from pipelines.training.seq_trainer import SequenceTrainer
from modules.decision_transformer import DecisionTransformer

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


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
    act_dim = np.squeeze(sequences[0]['actions'].shape[1:])
    state_dim = np.squeeze(sequences[0]['states'].shape[1:])
    max_ep_len = max([len(path['states']) for path in sequences])  # take it as the longest trajectory
    scale = np.mean([len(path['states']) for path in sequences])  # scale for rtg

    # train-eval split
    eval_size = config.get('eval_size', 500)
    eval_sequences = sequences[:eval_size]
    sequences = sequences[eval_size:]

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
    input_type = config.get('input_type', 'coord')
    states = np.concatenate(states, axis=0)
    if input_type == 'grayscale':
        # no normalization needed for cnn
        state_mean, state_std = np.array(50), np.array(100)
    else:
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        state_mean[[0,5,10,15,20]] = 0
        state_std[[0,5,10,15,20]] = 1
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {config["experiment_name"]}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = config['context_length']
    batch_size = config['batch_size']
    num_eval_episodes = config['num_eval_episodes']
    pct_traj = config['pct_traj']

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest return
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(sequences) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K, eval=False):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        ) if not eval else np.arange(len(eval_sequences))

        if eval:
            batch_size = len(eval_sequences)

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = sequences[int(sorted_inds[batch_inds[i]])] if not eval else eval_sequences[batch_inds[i]]
            si = random.randint(1, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['states'][max(si - max_len, 0):si].reshape(1, -1, state_dim))
            a.append(traj['actions'][max(si - max_len, 0):si].reshape(1, -1, act_dim))
            r.append(traj['rewards'][max(si - max_len, 0):si].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][max(si - max_len, 0):si].reshape(1, -1))
            else:
                d.append(traj['dones'][max(si - max_len, 0):si].reshape(1, -1))
            timesteps.append(np.arange(max(si - max_len, 0), max(si - max_len, 0) + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(
                discount_cumsum(traj['rewards'][max(si - max_len, 0):si], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1,
                                                                                                                 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = (s[-1] - state_mean) / state_std
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0, a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask
    # def get_batch(batch_size=256, max_len=K):
    #     batch_inds = np.random.choice(
    #         np.arange(num_trajectories),
    #         size=batch_size,
    #         replace=True,
    #         p=p_sample,  # reweights so we sample according to timesteps
    #     )
    #
    #     s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    #     for i in range(batch_size):
    #         traj = sequences[int(sorted_inds[batch_inds[i]])]
    #         si = random.randint(0, traj['rewards'].shape[0] - 1)
    #
    #         # get sequences from dataset
    #         s.append(traj['states'][si:si + max_len].reshape(1, -1, state_dim))
    #         a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
    #         r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
    #         if 'terminals' in traj:
    #             d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
    #         else:
    #             d.append(traj['dones'][si:si + max_len].reshape(1, -1))
    #         timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
    #         timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
    #         rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
    #         if rtg[-1].shape[1] <= s[-1].shape[1]:
    #             rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
    #
    #         # padding and state + reward normalization
    #         tlen = s[-1].shape[1]
    #         s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
    #         s[-1] = (s[-1] - state_mean) / state_std
    #         a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
    #         r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
    #         d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
    #         rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
    #         timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
    #         mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
    #
    #     s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    #     a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    #     r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    #     d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    #     rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    #     timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    #     mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
    #
    #     return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths, crashes = [], [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length, crash = evaluate_episode_rtg(
                        config['env'],
                        state_dim,
                        act_dim,
                        model,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        target_return=target_rew,
                        render=config['eval_render'],)
                returns.append(ret)
                lengths.append(length)
                crashes.append(crash)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_not_crashed (out of {num_eval_episodes} runs)': np.sum(crashes),
            }
        return fn

    print("state_dim:", state_dim, " act_dim:", act_dim, " K:", K, " max_ep_len:", max_ep_len, " scale:", scale)

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
            project='highway-decision-transformer',
            config={}#config
        )
        # wandb.watch(model)  # wandb has some bug

    count_parameters(model)

    max_ret = 0
    for iteration in range(config['max_iters']):
        outputs = trainer.train_iteration(num_steps=config['num_steps_per_iter'], iter_num=iteration+1, print_logs=True)
        eval_rets = [outputs[f'evaluation/target_{target_rew}_return_mean'] for target_rew in config['env_targets']]
        mean_ret = np.mean(eval_rets)
        if mean_ret > max_ret:
            max_ret = mean_ret
            if config['log_highest_return']:
                print("Saving model with highest mean return so far", mean_ret)
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'saved_models/best-{round(max_ret, 2)}-checkpoint-{config["experiment_name"]}.pth')
        if iteration%10 == 0 and iteration > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'saved_models/iter-{iteration}-checkpoint-{config["experiment_name"]}.pth')

        if config['log_to_wandb']:
            wandb.log(outputs)

    return model, optimizer, scheduler


