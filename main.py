from models import PoseViT
from lsfb_dataset.datasets import LSFBIsolConfig, LSFBIsolLandmarks
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from schedulers import WarmupLinearScheduler
import math
from training import ClassificationTrainer
from transforms import MergeLandmarks, FlattenLandmarks
from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transform import Padding
from torchmetrics import Accuracy, F1Score

n_labels = 2000
n_epochs = 100
seq_size = 30
warmups_steps = math.floor(n_epochs * 0.2)

DS_ROOT = "/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/isol"

# Creating the transforms for test and train

merge = MergeLandmarks(
    {"pose": [0, 23], "left_hand": [None, None], "right_hand": [None, None]}
)
padding = Padding(seq_size)
flatten = FlattenLandmarks()

train_transforms = Compose([merge, padding, flatten])
test_transforms = Compose([merge, padding, flatten])

# Configuring the training and testing splits

train_config = LSFBIsolConfig(
    root=DS_ROOT,
    split="train",
    n_labels=n_labels,
    landmarks=["pose", "left_hand", "right_hand"],
    sequence_max_length=seq_size,
    transform=train_transforms,
)


test_config = LSFBIsolConfig(
    root=DS_ROOT,
    split="test",
    n_labels=n_labels,
    landmarks=["pose", "left_hand", "right_hand"],
    sequence_max_length=seq_size,
    transform=test_transforms,
)


# Creation landmarks loader
train_dataset = LSFBIsolLandmarks(train_config)
test_dataset = LSFBIsolLandmarks(test_config)

# Create data loader
data = {}

data["train"] = DataLoader(train_dataset, batch_size=128, shuffle=True)
data["test"] = DataLoader(test_dataset, batch_size=128, shuffle=True)


# Create model
model = PoseViT(n_labels, embedding_size=128, seq_size=seq_size)


# Criterion
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# Scheduler
scheduler = WarmupLinearScheduler(optimizer, warmups_steps, n_epochs)


# Create trainer
trainer = ClassificationTrainer(
    data,
    model,
    optimizer,
    criterion,
    scheduler,
)

# Accuracy
trainer.add_train_metric("accuracy", Accuracy(task="multiclass", num_classes=n_labels))
trainer.add_test_metric("accuracy", Accuracy(task="multiclass", num_classes=n_labels))

# Top 10 accuracy
trainer.add_train_metric(
    "top-10 accuracy", Accuracy(task="multiclass", num_classes=n_labels, top_k=10)
)
trainer.add_test_metric(
    "top-10 accuracy", Accuracy(task="multiclass", num_classes=n_labels, top_k=10)
)

# Balanced accuracy
trainer.add_train_metric(
    "balanced accuracy",
    Accuracy(task="multiclass", num_classes=n_labels, average="macro"),
)

trainer.add_test_metric(
    "balanced accuracy",
    Accuracy(task="multiclass", num_classes=n_labels, average="macro"),
)


# Balanced accuracy
trainer.add_train_metric(
    "balanced top-10 accuracy",
    Accuracy(task="multiclass", num_classes=n_labels, average="macro", top_k=10),
)

trainer.add_test_metric(
    "balanced top-10 accuracy",
    Accuracy(task="multiclass", num_classes=n_labels, average="macro", top_k=10),
)

# F1 score

trainer.add_train_metric("f1-score", F1Score(task="multiclass", num_classes=n_labels))
trainer.add_test_metric("f1-score", F1Score(task="multiclass", num_classes=n_labels))


# Train
trainer.fit(n_epochs)
