class SplitLandmarks:
    def __init__(self, config):
        """
        Merge the landmarks into a single array

        Args:
            config: A dictionnary with a key for each landmark to merge and a start and end value for each landmark
            e.g. {"pose": [0, 23], "left_hand": [23, 42], "right_hand": [42, 61]}

        """

        self.config = config

        pass

    def __call__(self, x: dict) -> dict:
        # get landmarks:
        landmarks = {}

        for key in self.config.keys():
            start, end = self.config[key]
            landmarks[key] = x[:, start:end, :]

        return landmarks
