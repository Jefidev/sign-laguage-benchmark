from abc import abstractmethod


class Metric:
    """Metric used in trainer classes.

    Args:
        name: The name of the metric.
        printable: If True, the metric is printed in trainer classes. Otherwise, it is not.
    """

    def __init__(self, name: str, printable: bool = True):
        self.name = name
        self.printable = printable
        self.history = []

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def better_than(self, given_value):
        """
        Indicate if the current value of the metric is better than a given value.
        Only values of the same metrics can be compared.

        The default behavior is to return false

        Args:
            given_value: A given value of the metric
        """
        return False

    def next(self, targets, logits, predictions, loss):
        """
        Called at each prediction of the model

        Args:
            targets: The targets of the current instance
            logits: The raw output of the model
            predictions: The predicted label for the current instance
            loss: The loss of the prediction
        """
        pass

    def save(self):
        """
        Save the metric and its history.
        """
        print(f"{self.name} does not have save function.")

    def __call__(self, *args, **kwargs):
        """See self.next(...)"""
        self.next(*args, **kwargs)
