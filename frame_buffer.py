import torchvision.transforms.v2 as transforms
import numpy as np


def process_frame(frame, hw):
    gray_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.CenterCrop((175, 150)),
        transforms.Resize((hw, hw)),
        transforms.ToImageTensor()
    ])
    frame = gray_transform(frame).numpy() / 255
    return frame


class FrameBuffer:
    def __init__(self, max_size=2, alpha=0.4, hw=84):
        self._buffer = []  # memory for last frames
        self.max_size = max_size  # max size of memory
        self._alpha = alpha  # discount factor for oldest memories
        self.hw = hw  # height and width of frames

    def add_raw_frame(self, frame):
        frame = process_frame(frame, self.hw)
        self.add_frame(frame)

    def add_frame(self, frame):
        if len(self._buffer) == self.max_size:
            self._buffer.pop()
        self._buffer.insert(0, frame)

    def get_image(self):
        img = self._buffer[0].copy()
        alpha = self._alpha
        for i in range(1, len(self._buffer)):
            img += alpha * self._buffer[i]
            alpha *= self._alpha
        img = np.clip(img, 0.0, 1.0)
        return img

    def clear(self):
        self._buffer.clear()
