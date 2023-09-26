import torch
import matplotlib.pyplot as plt
from .metric import Metric


class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__("Accuracy")
        self.total = 0
        self.correct = 0

    def value(self):
        return self.correct / self.total

    def better_than(self, given_value):
        return given_value < self.value()

    def next(self, targets, scores, predictions, loss):
        self.total += torch.numel(targets)
        # noinspection PyTypeChecker
        self.correct += torch.sum(targets == predictions).item()

    def reset(self):
        self.total = 0
        self.correct = 0

    def plot(self):
        plt.figure()
        plt.title("Accuracy")
        plt.plot(self.history)
        plt.show()
