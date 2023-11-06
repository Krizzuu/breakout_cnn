import time
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
# import cv2 as cv
import pandas as pd


def process_frame(frame, n_frame=None, hw=84, alpha=0.4):
    gray_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.CenterCrop((175, 150)),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    frame = gray_transform(frame) / 255

    if n_frame is not None:
        n_frame = gray_transform(n_frame) / 255
        new_frame = n_frame - frame * alpha
    else:
        new_frame = frame

    return new_frame.numpy()


def __debug_states(states):
    for i in range(states.shape[0]):
        save_image(states[i], f"./states/{i}.png")
    #     np_img = states[i].detach().cpu().numpy().reshape(84, 84)
    #     cv.imwrite(f"./states/{i}.png", np_img * 255)



class ValueNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 filename,
                 conv_layers=((8, 3, 1, 0),),
                 fc_dims=(512, 64,)
                 ):
        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        # conv_layers=((16, 8, 4, 2), (32, 4, 2, 1),),
        # conv_layers=((8, 3, 1, 0),),

        # fc_dims=(4096, 512,)

        # Convolutional layers

        kernels, kernel_size, stride, padding = conv_layers[0]
        self.l1 = nn.Sequential()
        self.l1.append(nn.Conv2d(1, kernels, kernel_size=kernel_size, stride=stride, padding=padding))
        self.l1.append(nn.ReLU())

        prev_props = (1, 1, 1)

        for i, conv_props in enumerate(conv_layers):
            kernels, kernel_size, stride, padding = conv_props
            if i > 0:
                conv = nn.Conv2d(prev_props[0], kernels, kernel_size=kernel_size, stride=stride, padding=padding)
                self.l1.append(conv)
                self.l1.append(nn.ReLU())
            prev_props = conv_props

        self.l1.append(nn.MaxPool2d(2, 2))
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

    def save_model(self):
        torch.save(self.state_dict(), './models/' + self.filename + '.pth')

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
                 training_strategy,
                 evaluation_strategy,
                 replay_buffer,
                 filename="breakout_cnn",
                 hw=84):
        self.hw = hw
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.min_batches_to_update = min_batches_to_update
        self.replace_target_n = replace_target_n

        self.online_model = ValueNetwork(self.state_space, self.action_space, filename)
        self.target_model = ValueNetwork(self.state_space, self.action_space, filename + "_target")

        try:
            self.online_model.load_model()
            print('loaded pretrained model')
        except:
            pass

        self._target_replace_cnt = 0

        # set the same target as online model
        self.update_target()

        self.training_strategy = training_strategy
        self.evaluation_strategy = evaluation_strategy

        self.replay_buffer = replay_buffer

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")

        self.device = torch.device(device)

        self.loss = nn.SmoothL1Loss()
        self.value_optimizer = optim.Adam(self.online_model.parameters(), lr=lr)

    def update_target(self):
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences

        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        value_loss = self.loss(q_sa, target_q_sa).to(self.device)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, n_episodes=100, render=False):
        max_score = 0
        scores = np.zeros(n_episodes)
        avg_scores = np.zeros(n_episodes)
        std_scores = np.zeros(n_episodes)
        times = []

        training_start = time.time()
        min_samples = self.min_batches_to_update * self.replay_buffer.batch_size

        for e in range(n_episodes):
            was_target_replaced = False
            score = 0
            done = False

            raw_state, i = self.env.reset()

            episode_start = time.time()

            step = 0
            
            state = process_frame(raw_state)
            for step in count():

                action = self.training_strategy.select_action(self.online_model, state)
                raw_next_state, reward, done, is_truncated, i = self.env.step(action)
                next_state = process_frame(raw_state, raw_next_state)
                if render:
                    self.env.render()

                experience = (
                    state.reshape(1, 1, self.hw, self.hw),
                    action,
                    reward,
                    next_state.reshape(1, 1, self.hw, self.hw),
                    done,
                    raw_next_state
                )
                self.replay_buffer.store(experience)

                # Update
                if len(self.replay_buffer) >= min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)
                    self._target_replace_cnt += 1
                    # print("Online updated !")
                    if self._target_replace_cnt > self.replace_target_n:
                        self._target_replace_cnt = 0
                        self.update_target()
                        was_target_replaced = True

                score += reward
                if done:
                    break
                state = next_state
                raw_state = raw_next_state
                # time.sleep(0.002)

            episode_time = (time.time() - episode_start) / 60
            times.append(episode_time)

            min_score_idx100 = max(e + 1 - 100, 0)

            if score > max_score:
                max_score = score
            scores[e] = score
            avg_scores[e] = np.mean(scores[min_score_idx100:e + 1])
            std_scores[e] = np.std(scores[min_score_idx100:e + 1])

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
            #             was_target_replaced = True
            #             # print("Target updated !")

            print(f"Episode {e:5}: Score : {int(score):3}, Steps: {step}, "
                  f"Avg score (last 100): {avg_scores[e]:.2} Max: {max_score} "
                  f"eps: {self.training_strategy.epsilon:.2} {'Target replaced' if was_target_replaced else self._target_replace_cnt}")

            if e > 0 and e % 200 == 0:
                self.online_model.save_model()
                df = pd.DataFrame(data=np.array([avg_scores[:e + 1], std_scores[:e + 1]]).T,
                                  index=np.arange(e + 1),
                                  columns=['avg100', 'std100'])
                df.to_csv("last_run.csv")


