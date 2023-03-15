# trainder module adopted from official code of Decision Transformer
# https://github.com/kzl/decision-transformer
# @article{chen2021decisiontransformer,
#   title={Decision Transformer: Reinforcement Learning via Sequence Modeling},
#   author={Lili Chen and Kevin Lu and Aravind Rajeswaran and Kimin Lee and Aditya Grover and Michael Laskin and Pieter Abbeel and Aravind Srinivas and Igor Mordatch},
#   journal={arXiv preprint arXiv:2106.01345},
#   year={2021}
# }
# MIT License
#
# Copyright (c) 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors (https://arxiv.org/abs/2106.01345)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import torch

from pipelines.training.trainer import Trainer


class ActTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
