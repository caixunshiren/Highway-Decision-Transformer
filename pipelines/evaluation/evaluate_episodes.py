import numpy as np
import torch
import gymnasium as gym
import highway_env


def evaluate_episode(
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):
    #TODO @jwoo implement this function for highway env
    raise NotImplementedError

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    highway_env.register_highway_envs()

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        render=True,
        sample=False,
        scale=60,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    (state, info), done = env.reset(), False

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length, crash = [], [], []

    state, info = env.reset()
    done = truncated = False
    episode_return, episode_length = 0, 0
    t = 0
    while not (done or truncated):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        if sample:
            distribution = torch.distributions.Categorical(logits=torch.exp(torch.tensor(action)))
            optimal_action = distribution.sample()
        else:
            optimal_action = np.argmax(action)

        state, reward, done, truncated, info = env.step(optimal_action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # render the highway environment
        if render:
            env.render()

        pred_return = target_return[0, -1] - (reward/scale)

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        t += 1

    crash = truncated
    env.close()

    return episode_return, episode_length, crash


def evaluate_episode_rtg_nonstop(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        render=True,
        sample=False,
        scale=60,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    (state, info), done = env.reset(), False

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length, crash = [], [], []

    state, info = env.reset()
    done = truncated = False
    episode_return, episode_length = 0, 0
    t = 0
    while not (done or truncated):

        # if reach max episode length, take away half of past states, actions, rewards
        if t == max_ep_len-1:
            start_idx = int(max_ep_len/2)
            actions = actions[start_idx:]
            rewards = rewards[start_idx:]
            states = states[start_idx:]
            timesteps = timesteps[:, start_idx:].reshape(1, -1)
            target_return = target_return[:, start_idx:].reshape(1, -1)
            t = start_idx

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        if sample:
            distribution = torch.distributions.Categorical(logits=torch.exp(torch.tensor(action)))
            optimal_action = distribution.sample()
        else:
            optimal_action = np.argmax(action)

        state, reward, done, truncated, info = env.step(optimal_action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # render the highway environment
        if render:
            env.render()

        pred_return = target_return[0, -1] - (reward/scale)

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        t += 1

    crash = truncated
    env.close()

    return episode_return, episode_length, crash
