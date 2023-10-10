import gymnasium as gym

from frame_buffer import FrameBuffer, process_frame
from model import DQN
from strategy import EpsGreedyExpStrategy, GreedyStrategy

env_name = 'ALE/Breakout-v5'
gamma = 0.99
episodes = 80000
min_batches_to_update = 4
replace_target_n = 480
hw = 84

if __name__ == '__main__':
    # getting basic information about environment
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    action_space = env.action_space.n
    (raw_s, _) = env.reset()


    # preparing tools
    training_strategy_fn = lambda: EpsGreedyExpStrategy()
    evaluation_strategy_fn = lambda: GreedyStrategy()
    frame_buffer_fn = lambda: FrameBuffer(hw=hw)

    s = process_frame(raw_s, hw)
    state_space = s.shape

    # initializing agent
    agent = DQN(
        env,
        state_space,
        action_space,
        gamma,
        lr=0.01,
        min_batches_to_update=min_batches_to_update,
        replace_target_n=replace_target_n,
        training_strategy_fn=training_strategy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn,
        frame_buffer_fn=frame_buffer_fn
    )

    agent.train(n_episodes=episodes, render=False)





