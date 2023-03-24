import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
import numpy as np
# from utils import RecordVideo
import torch
from modules.decision_transformer import DecisionTransformer
import pickle


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
A = np.load('../data/mcts_dataset_expert.npy', allow_pickle=True)
sequences = [load_sequence(row) for row in A]

# load model
checkpoint = torch.load('saved_models/checkpoint-mlp-decision-transformer-expert-mcts.pth', map_location='cpu', pickle_module=pickle)
config = {
    'device': 'cpu',#'cuda',
    'mode': 'normal',
    'experiment_name': 'mlp-decision-transformer-expert-mcts',
    'group_name': 'ECE324',
    'log_to_wandb': False,
    'max_iters': 10,
    'num_steps_per_iter': 1000,#10000,
    'context_length': 30,
    'batch_size': 32,
    'num_eval_episodes': 50,
    'pct_traj': 1.0,
    'n_layer': 3,
    'embed_dim': 128,
    'n_head': 4,
    'activation_function': 'relu',
    'dropout': 0.1,
    'model': checkpoint['model'],
    'optimizer': checkpoint['optimizer'],
    'learning_rate': 1e-4,
    'warmup_steps': 1000,#10000,
    'weight_decay': 1e-4,
    'env_targets': [],#[0.5, 1.0, 5.0, 10],
    'action_tanh': False, #True,
    'loss_fn': lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, torch.argmax(a, dim=1)),
    #lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    'err_fn': lambda a_hat, a: torch.sum(torch.argmax(a_hat, dim=1) != torch.argmax(a, dim=1))/a.shape[0],
}

env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.config["duration"] = 59


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


highway_env.register_highway_envs()


model.eval()
model.to(device=config['device'])

state_mean = torch.from_numpy(state_mean).to(device=config['device'])
state_std = torch.from_numpy(state_std).to(device=config['device'])


# we keep all the histories on the device
# note that the latest action and reward will be "padding"
states = torch.from_numpy(state).reshape(1, state_dim).to(device=config['device'], dtype=torch.float32)


actions = torch.zeros((0, act_dim), device=config['device'], dtype=torch.float32)
rewards = torch.zeros(0, device=config['device'], dtype=torch.float32)

target_return = 60.0

ep_return = target_return
target_return = torch.tensor(ep_return, device=config['device'], dtype=torch.float32).reshape(1, 1)
timesteps = torch.tensor(0, device=config['device'], dtype=torch.long).reshape(1, 1)

sim_states = []
for episodes in range(10):
    state, info = env.reset()
    done = truncated = False
    episode_return, episode_length = 0, 0
    t = 0
    while not (done or truncated):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=config['device'])], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=config['device'])])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        print(np.argmax(action), action, t)

        optimal_action = np.argmax(action)

        state, reward, done, truncated, info = env.step(optimal_action)

        cur_state = torch.from_numpy(state).to(device=config['device']).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # render the highway environment
        env.render()


        pred_return = target_return[0, -1]

        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=config['device'], dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        t+=1

env.close()