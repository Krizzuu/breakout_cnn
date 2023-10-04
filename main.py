import gymnasium as gym
import cv2 as cv

from model import CnnModel
from replay_buffer import ReplayBuffer

env_settings = {
    'env_name': 'ALE/Breakout-v5',
    'gamma': 1.00,
    'max_minutes': 10,
    'max_episodes': 10000,
    'goal_mean_100_reward': 475
}


def process_frame(frame, xy=64):
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # remove irrelevant parts - scoreboard, lifes
    frame = frame[31:200, 7:155]
    frame = cv.resize(frame, (xy, xy), interpolation=cv.INTER_NEAREST)
    cv.imwrite("./debug_img/test.png", frame)
    frame = frame.reshape(1, 1, xy, xy)
    return frame


model = CnnModel()
replay_buffer = ReplayBuffer()

if __name__ == '__main__':
    env = gym.make(env_settings['env_name'], render_mode="human")
    for _ in range(3):  # run for 3 episodes
        frame, _ = env.reset()
        frame = process_frame(frame)
        done = False
        while not done:
            env.render()
            # action = env.action_space.sample()
            # action = model.forward(state)  # TODO
            action = 1
            new_frame, reward, done, truncated, info = env.step(action)
            new_frame = process_frame(new_frame)
            failed = done and not truncated
            experience = (frame, action, reward, new_frame, float(failed))
            replay_buffer.store(experience)
            frame = new_frame


