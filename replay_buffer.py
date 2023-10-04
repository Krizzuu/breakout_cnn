import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=10000, batch_size=64):
        self.frame_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # state
        self.award_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # action
        self.reward_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # reward
        self.next_frame_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # next state
        self.done_mem = np.empty(shape=(max_size), dtype=np.ndarray)  # is done

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        s, a, r, p, d = sample
        self.frame_mem[self._idx] = s
        self.award_mem[self._idx] = a
        self.reward_mem[self._idx] = r
        self.next_frame_mem[self._idx] = p
        self.done_mem[self._idx] = d

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.frame_mem[idxs]), \
            np.vstack(self.award_mem[idxs]), \
            np.vstack(self.reward_mem[idxs]), \
            np.vstack(self.next_frame_mem[idxs]), \
            np.vstack(self.done_mem[idxs])
        return experiences

    def __len__(self):
        return self.size
