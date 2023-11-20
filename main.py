import gymnasium as gym

from model import DQN, process_frame
from strategy import EpsGreedyExpStrategy, GreedyStrategy, EpsGreedyLinearStrategy, EpsGreedyStrategy
from replay_buffer import ReplayBuffer

# env_name = 'ALE/Breakout-v5'
# env_name = 'BreakoutNoFrameskip-v4'
# env_name = 'BreakoutDeterministic-v4'
env_name = 'Breakout-v4'
gamma = 0.99
episodes = 80000
min_batches_to_update = 4
replace_target_n = 10000
hw = 84
lr = 0.001

if __name__ == '__main__':
    # getting basic information about environment
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    action_space = env.action_space.n
    (raw_s, _) = env.reset()

    # preparing tools
    training_strategy = EpsGreedyLinearStrategy(init_epsilon=0.7, min_epsilon=0.05, decay_steps=250000)
    # training_strategy = EpsGreedyStrategy(0.05)
    evaluation_strategy = GreedyStrategy()
    replay_buffer = ReplayBuffer(max_size=40000)

    s = process_frame(raw_s)
    state_space = s.shape

    print("Creating agent")

    # initializing agent
    agent = DQN(
        env,
        state_space,
        action_space,
        gamma,
        lr=lr,
        min_batches_to_update=min_batches_to_update,
        replace_target_n=replace_target_n,
        training_strategy=training_strategy,
        evaluation_strategy=evaluation_strategy,
        replay_buffer=replay_buffer
    )

    print("Starting training")

    agent.train(n_episodes=episodes, render=False)





