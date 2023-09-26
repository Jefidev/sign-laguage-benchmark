from metrics import Accuracy
import torch


def test_accuracy():
    accuracy = Accuracy()

    target = torch.tensor([1, 1, 1, 1])
    prediction = torch.tensor([0, 0, 0, 0])

    # Test value 0
    accuracy.next(target, None, prediction, None)
    assert accuracy.value() == 0.0

    # test next don't reset
    target = torch.tensor([1, 1, 1, 1])
    prediction = torch.tensor([1, 1, 1, 1])
    accuracy.next(target, None, prediction, None)
    assert accuracy.value() == 0.5

    # Test reset actually reset
    accuracy.reset()
    accuracy.next(target, None, prediction, None)
    assert accuracy.value() == 1.0
