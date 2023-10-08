import gymnasium as gym

from model import DQN
from strategy import EpsGreedyExpStrategy, GreedyStrategy

env_name = 'ALE/Breakout-v5'
gamma = 0.99
episodes = 1000
min_batches_to_update = 4
replace_target_n = 4

if __name__ == '__main__':
    # getting basic information about environment
    env = gym.make(env_name, render_mode="human")
    action_space = env.action_space.n
    (s, _) = env.reset()
    state_space = s.shape
    state_space = (state_space[2], state_space[0], state_space[1])

    # preparing tools
    training_strategy_fn = lambda: EpsGreedyExpStrategy()
    evaluation_strategy_fn = lambda: GreedyStrategy()

    # initializing agent
    agent = DQN(
        env,
        state_space,
        action_space,
        gamma,
        lr=0.0005,
        min_batches_to_update=min_batches_to_update,
        replace_target_n=replace_target_n,
        training_strategy_fn=training_strategy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn
    )





