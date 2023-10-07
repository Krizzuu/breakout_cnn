import gymnasium as gym



env_name=  'ALE/Breakout-v5'
gamma = 1.00
episodes = 1000


if __name__ == '__main__':
    env = gym.make(env_name, render_mode="human")
    action_space = env.action_space.n
    (s, _) = env.reset()
    state_space = s.shape
    state_space = (state_space[2], state_space[0], state_space[1])



