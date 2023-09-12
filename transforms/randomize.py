import numpy as np


class Randomize:
    def __init__(self, transform, odds=0.5):
        self.transform = transform
        self.odds = odds

    def __call__(self, x: np.array) -> np.array:
        if np.random.rand() < self.odds:
            return self.transform(x)
        else:
            return x
