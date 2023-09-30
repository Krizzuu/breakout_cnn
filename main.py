import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

env_settings = {
    'env_name': 'ALE/Breakout-v5',
    'gamma': 1.00,
    'max_minutes': 10,
    'max_episodes': 10000,
    'goal_mean_100_reward': 475
}

if __name__ == '__main__':
    env = gym.make(env_settings['env_name'], render_mode="human")
    for _ in range(3):  # run for 3 episodes
        state, _ = env.reset()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
    pass

