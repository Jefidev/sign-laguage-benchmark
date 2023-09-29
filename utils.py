from transforms import get_skeleton_transforms, get_data_augmentation_transforms
from lsfb_dataset.datasets import LSFBIsolConfig, LSFBIsolLandmarks
from torchmetrics import Accuracy, F1Score
import wandb
import torch


def load_datasets(path, seq_size, n_labels, data_augmentation=False, dry_run=False):
    if data_augmentation:
        train_transforms = get_data_augmentation_transforms(seq_size)
    else:
        train_transforms = get_skeleton_transforms(seq_size)

    test_transforms = get_skeleton_transforms(seq_size)

    if dry_run:
        train, test = "mini_sample", "mini_sample"
    else:
        train, test = "train", "test"

    train_config = LSFBIsolConfig(
        root=path,
        split=train,
        n_labels=n_labels,
        landmarks=["pose", "left_hand", "right_hand"],
        sequence_max_length=seq_size,
        transform=train_transforms,
    )

    test_config = LSFBIsolConfig(
        root=path,
        split=test,
        n_labels=n_labels,
        landmarks=["pose", "left_hand", "right_hand"],
        sequence_max_length=seq_size,
        transform=test_transforms,
    )

    train_dataset = LSFBIsolLandmarks(train_config)
    test_dataset = LSFBIsolLandmarks(test_config)

    return train_dataset, test_dataset


def set_common_metrics(trainer, n_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accuracy
    trainer.add_train_metric(
        "accuracy", Accuracy(task="multiclass", num_classes=n_labels).to(device)
    )
    trainer.add_test_metric(
        "accuracy", Accuracy(task="multiclass", num_classes=n_labels).to(device)
    )

    # Top 10 accuracy
    trainer.add_train_metric(
        "top-10 accuracy",
        Accuracy(task="multiclass", num_classes=n_labels, top_k=10).to(device),
    )
    trainer.add_test_metric(
        "top-10 accuracy",
        Accuracy(task="multiclass", num_classes=n_labels, top_k=10).to(device),
    )

    # Balanced accuracy
    trainer.add_train_metric(
        "balanced accuracy",
        Accuracy(task="multiclass", num_classes=n_labels, average="macro").to(device),
    )

    trainer.add_test_metric(
        "balanced accuracy",
        Accuracy(task="multiclass", num_classes=n_labels, average="macro").to(device),
    )

    # Balanced accuracy
    trainer.add_train_metric(
        "balanced top-10 accuracy",
        Accuracy(task="multiclass", num_classes=n_labels, average="macro", top_k=10).to(
            device
        ),
    )

    trainer.add_test_metric(
        "balanced top-10 accuracy",
        Accuracy(task="multiclass", num_classes=n_labels, average="macro", top_k=10).to(
            device
        ),
    )

    # F1 score

    trainer.add_train_metric(
        "f1-score", F1Score(task="multiclass", num_classes=n_labels).to(device)
    )
    trainer.add_test_metric(
        "f1-score", F1Score(task="multiclass", num_classes=n_labels).to(device)
    )
    pass


def log_metrics(train_metrics, test_metrics):
    for metric_name, metric_value in train_metrics:
        wandb.log({"train " + metric_name: metric_value.compute()})

    for metric_name, metric_value in test_metrics:
        wandb.log({"valid " + metric_name: metric_value.compute()})


def log_losses(train_loss, test_loss):
    wandb.log({"train loss": train_loss})
    wandb.log({"valid loss": test_loss})


def log_confusion_matrix(confusion_matrix, labels):
    # TODO
    pass
