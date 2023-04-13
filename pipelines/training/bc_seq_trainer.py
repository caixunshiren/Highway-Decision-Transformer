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


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        #print("debug action pred and target before: ", action_preds.shape, action_target.shape)

        act_dim = action_preds.shape[2]
        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        #print("debug action pred and target: ", action_preds.shape, action_target.shape,
        #      "debug state pred and states: ", state_preds.shape, states.shape)

        #print("debug action pred and target: ", action_preds, action_target)
        #print("debug target index: ", torch.argmax(action_target, dim=1))
        
        action_target = action_target[:,-1:,:].reshape(-1, 1, action_target.shape[2])

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target[:,-1:,:], None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            if self.err_fn is not None:
                self.diagnostics['training/action_error'] = self.err_fn(action_preds, action_target)\
                    .detach().cpu().item()
            else:
                self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2)\
                    .detach().cpu().item()

        return loss.detach().cpu().item()
