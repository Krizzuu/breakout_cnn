import gymnasium as gym

from frame_buffer import FrameBuffer
from replay_buffer import ReplayBuffer

env_settings = {
    'env_name': 'ALE/Breakout-v5',
    'gamma': 1.00,
    'max_minutes': 10,
    'max_episodes': 10000,
    'goal_mean_100_reward': 475
}

frame_buffer = FrameBuffer()
replay_buffer = ReplayBuffer()

if __name__ == '__main__':
    env = gym.make(env_settings['env_name'], render_mode="human")
    for _ in range(3):  # run for 3 episodes
        state, _ = env.reset()
        done = False
        while not done:
            env.render()
            action = 1
            new_state, reward, done, truncated, info = env.step(action)
            frame_buffer.add_frame(new_state)
            i = frame_buffer.get_image()
            failed = done and not truncated
            experience = (state, action, reward, new_state, float(failed))
            replay_buffer.store(experience)
            state = new_state


