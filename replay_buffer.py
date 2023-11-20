import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=10000, batch_size=64):
        self.state_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # state
        self.action_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # action
        self.reward_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # reward
        self.next_state_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # next state
        self.done_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # is done
        self.raw_frame_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # raw frame

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        s, a, r, p, d, rs = sample
        self.state_mem[self._idx] = s
        self.action_mem[self._idx] = a
        self.reward_mem[self._idx] = r
        self.next_state_mem[self._idx] = p
        self.done_mem[self._idx] = d
        # self.raw_frame_mem[self._idx] = rs

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.state_mem[idxs]), \
            np.vstack(self.action_mem[idxs]), \
            np.vstack(self.reward_mem[idxs]), \
            np.vstack(self.next_state_mem[idxs]), \
            np.vstack(self.done_mem[idxs])
        return experiences

    def __len__(self):
        return self.size
