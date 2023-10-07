import torchvision.transforms.v2 as transforms
import numpy as np


def process_frame(frame):
    gray_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.CenterCrop((175, 150)),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    frame = gray_transform(frame).numpy()
    return frame


class FrameBuffer:
    def __init__(self, max_size=2, alpha=0.4):
        self._buffer = np.zeros((3, 84, 84))
        self.max_size = max_size
        self._alpha = alpha
        self._idx = 0

    def add_frame(self, frame):
        if self._idx > 0:
            self._buffer[2] = self._buffer[1]
            self._buffer[1] = self._buffer[0]
        frame = process_frame(frame)
        self._buffer[0] = frame
        self._idx += 1

    def get_image(self):
        img = np.zeros_like(self._buffer[0])
        alpha = self._alpha
        for i in range(self._idx):
            img += alpha * self._buffer[i]
            alpha *= self._alpha
        img = np.clip(img, 0.0, 1.0)
        return img
