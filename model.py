import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim

from frame_buffer import FrameBuffer
from replay_buffer import ReplayBuffer


class ValueNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 filename,
                 conv_layers=((32, 8, 4), (64, 4, 2), (64, 3, 1)),
                 fc_dims=(512,)
                 ):
        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        # Convolutional layers

        self.l1 = nn.Sequential()
        self.l1.append(nn.Conv2d(1, conv_layers[0][0], conv_layers[0][1], conv_layers[0][2]))
        self.l1.append(nn.ReLU())

        prev_props = (1, 1, 1)

        for i, conv_props in enumerate(conv_layers):
            if i > 0:
                conv = nn.Conv2d(prev_props[0], conv_props[0], conv_props[1], conv_props[2])
                self.l1.append(conv)
                self.l1.append(nn.ReLU())
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
        x = self.l1(x)
        x = self.l2(x)
        x = self.output_layer(x)

        return x

    def save_model(self):
        torch.save(self.state_dict(), './models/' + self.filename + '.pth')

    def load_model(self):
        self.load_state_dict(torch.load('./models/' + self.filename + '.pth'))


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
                 filename="breakout"):
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.min_batches_to_update = min_batches_to_update
        self.replace_target_n = replace_target_n

        self.online_model = ValueNetwork(self.state_space, self.action_space, filename)
        self.target_model = ValueNetwork(self.state_space, self.action_space, filename+"_target")

        self.training_strategy = training_strategy_fn()
        self.evaluation_strategy = evaluation_strategy_fn()

        self.replay_buffer = ReplayBuffer()
        self.frame_buffer = FrameBuffer()

        self.online_updated = 0

        self.update_target()

        self.value_optimizer = optim.RMSprop(self.online_model.parameters(), lr=lr)

    def update_target(self):
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

