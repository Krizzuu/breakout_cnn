import time
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import cv2 as cv

from frame_buffer import process_frame
from replay_buffer import ReplayBuffer


class ValueNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 filename,
                 conv_layers=((32, 8, 4, 2), (64, 4, 2, 1), (64, 3, 1, 1)),
                 fc_dims=(512,)
                 ):
        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        # conv_layers = ((32, 8, 4, 2), (64, 4, 2, 1), (64, 3, 1, 1)),

        # Convolutional layers

        kernels, kernel_size, stride, padding = conv_layers[0]
        self.l1 = nn.Sequential()
        self.l1.append(nn.Conv2d(1, kernels, kernel_size=kernel_size, stride=stride, padding=padding))
        self.l1.append(nn.ReLU(inplace=True))

        prev_props = (1, 1, 1)

        for i, conv_props in enumerate(conv_layers):
            kernels, kernel_size, stride, padding = conv_props
            if i > 0:
                conv = nn.Conv2d(prev_props[0], kernels, kernel_size=kernel_size, stride=stride, padding=padding)
                self.l1.append(conv)
                self.l1.append(nn.ReLU(inplace=True))
            prev_props = conv_props

        self.l1.append(nn.Flatten())

        # Fully Connected layers
        self.l2 = nn.Sequential()
        fc_in = self.conv_output_dim()

        for i, dim in enumerate(fc_dims):
            fc = nn.Linear(fc_in, dim)
            fc_in = dim
            self.l2.append(fc)
            self.l2.append(nn.ReLU())

        # and the last Output Layer (also FC)
        self.output_layer = nn.Linear(fc_in, output_dim)

        # move to device if available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.to(self.device)

        # model name
        self.filename = filename

    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = self._format(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.output_layer(x)

        return x

    def save_model(self, note=""):
        torch.save(self.state_dict(), './models/' + self.filename + note + '.pth')

    def load_model(self):
        self.load_state_dict(torch.load('./models/' + self.filename + '.pth'))

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals

    def _format(self, _x):
        x = _x
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        elif x.device != self.device:
            x = x.clone().detach().to(self.device)
            x = x.unsqueeze(0)
        return x


class DQN:
    def __init__(self,
                 env,
                 state_space,
                 action_space,
                 gamma,
                 lr,
                 min_batches_to_update,
                 replace_target_n,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 frame_buffer_fn,
                 filename="breakout"):
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.min_batches_to_update = min_batches_to_update
        self.replace_target_n = replace_target_n

        self.online_model = ValueNetwork(self.state_space, self.action_space, filename)
        self.target_model = ValueNetwork(self.state_space, self.action_space, filename + "_target")
        self.target_model.eval()

        self.training_strategy = training_strategy_fn()
        self.evaluation_strategy = evaluation_strategy_fn()

        self.replay_buffer = ReplayBuffer()
        self.frame_buffer = frame_buffer_fn()

        self._target_replace_cnt = 0

        # set the same target as online model
        self.update_target()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.loss_criterion = nn.SmoothL1Loss()
        self.value_optimizer = optim.RMSprop(self.online_model.parameters(), lr=lr)

    def update_target(self):
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences

        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        # td_error = q_sa - target_q_sa
        # value_loss = self.loss_criterion(q_sa, target_q_sa).to(self.device)
        value_loss = self.loss_criterion(target_q_sa, q_sa).to(self.device)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, n_episodes=100, render=False):
        max_score = 0
        scores = np.zeros(n_episodes)
        times = []
        hw = self.frame_buffer.hw

        training_start = time.time()
        min_samples = self.min_batches_to_update * self.replay_buffer.batch_size

        for e in range(n_episodes):
            self.frame_buffer.clear()
            score = 0
            done = False

            raw_state, _ = self.env.reset()
            self.frame_buffer.add_raw_frame(raw_state)

            episode_start = time.time()

            step = 0
            for step in count():
                state = self.frame_buffer.get_image()
                action = self.training_strategy.select_action(self.online_model, state)
                raw_next_state, reward, done, is_truncated, _ = self.env.step(action)
                next_state = process_frame(raw_next_state, hw)
                self.frame_buffer.add_frame(next_state)
                if render:
                    self.env.render()

                experience = (state.reshape(1, 1, hw, hw), action, reward, next_state.reshape(1, 1, hw, hw), done)
                self.replay_buffer.store(experience)

                if len(self.replay_buffer) >= min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)
                    self._target_replace_cnt += 1
                    # print("Online updated !")
                    if self._target_replace_cnt > self.replace_target_n:
                        self._target_replace_cnt = 0
                        self.update_target()
                        # print("Target updated !")

                score += reward
                if done:
                    break


            episode_time = (time.time() - episode_start) / 60
            times.append(episode_time)

            if score > max_score:
                max_score = score
            scores[e] = score

            min_score_idx100 = max(e + 1 - 100, 0)
            print(f"Episode {e:5}: Score : {score}, time: {episode_time:6.2}, Steps: {step}, "
                  f"Avg score (last 100): {np.mean(scores[min_score_idx100:e + 1]):.2} Max: {max_score}")

            # Update
            # if len(self.replay_buffer) >= min_samples:
            #     n_optimizes = np.ceil(step / self.replay_buffer.batch_size).astype(int)
            #     for i in range(n_optimizes):
            #         experiences = self.replay_buffer.sample()
            #         experiences = self.online_model.load(experiences)
            #         self.optimize_model(experiences)
            #         self._target_replace_cnt += 1
            #         # print("Online updated !")
            #         if self._target_replace_cnt > self.replace_target_n:
            #             self._target_replace_cnt = 0
            #             self.update_target()
            #             # print("Target updated !")

            if e > 0 and e % self.replace_target_n == 0:
                self.target_model.save_model(str(e))
