import numpy as np
import torch
import torch.nn as nn
from modules.model import TrajectoryModel

class BehaviourCloning(TrajectoryModel):
    #CNN with n_layer hidden layers and 3 FC layers
    #Each hidden layer has hidden_size neurons
    #State dim is equivalent to input dim
    def __init__(self, input_dim, act_dim, hidden_size, n_layer, in_channels, channels, dropout=0.1, max_length=1, context_length=10, **kwargs):
        super().__init__(input_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [
            nn.Conv2d(in_channels * context_length, channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        for _ in range(n_layer-1):
            layers.extend([
                nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
        layers.extend([
            #The input dimension is reduced by 2 * number of pooling layers
            #The output dimension is the number of channels times the size of the output of the last pooling layer
            nn.Flatten(),
            nn.Linear(int(channels * input_dim/(2*(n_layer + 1))), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, rtg, timesteps, attention_mask=None, target_return=None):
        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat images
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)
        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, None, None, **kwargs)
        return actions[0,-1]