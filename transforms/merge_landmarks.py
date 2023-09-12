import numpy as np


class MergeLandmarks:
    def __init__(self, config):
        """
        Merge the landmarks into a single array

        Args:
            config: A dictionnary with a key for each landmark to merge and a start and end value for each landmark
            e.g. {"pose": [0, 23], "left_hand": [None, None], "right_hand": [None, None]}

        """

        self.config = config

        pass

    def __call__(self, x: dict) -> np.ndarray:
        # get landmarks:
        landmarks = []

        for key in self.config.keys():
            start, end = self.config[key]
            landmarks.append(x[key][:, start:end, :])

        pose = x["pose"][:, :23, :]

        # merge
        landmarks = np.concatenate(landmarks, axis=1)

        return landmarks
