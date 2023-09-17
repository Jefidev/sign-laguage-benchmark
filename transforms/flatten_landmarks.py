import numpy as np


class FlattenLandmarks:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        f, s, t = x.shape
        landmarks = x.reshape(f, s * t)

        return landmarks
