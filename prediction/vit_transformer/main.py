from training import ClassificationTrainer
from models import PoseViT
from lsfb_dataset.datasets import LSFBIsolConfig, LSFBIsolLandmarks
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from schedulers import WarmupLinearScheduler
import math

n_labels = 2000
n_epochs = 100
warmups_steps = math.floor(n_epochs * 0.2)


train_config = LSFBIsolConfig(
    root="/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/lsfb_isol",
    split="train",
    n_labels=n_labels,
    landmarks=["pose", "left_hand", "right_hand"],
    sequence_max_length=30,
)


test_config = LSFBIsolConfig(
    root="/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/lsfb_isol",
    split="test",
    n_labels=n_labels,
    landmarks=["pose", "left_hand", "right_hand"],
    sequence_max_length=30,
)


# Creation landmarks loader
train_dataset = LSFBIsolLandmarks(train_config)
test_dataset = LSFBIsolLandmarks(test_config)

# Create data loader
data = {}

data["train"] = DataLoader(train_dataset, batch_size=32, shuffle=True)
data["test"] = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Create model
model = PoseViT(
    n_labels,
    embedding_size=128,
    seq_size=len(train_dataset[0]),
)


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
    criterion,
    optimizer,
    scheduler,
)

# Add metrics TODO

# Train
trainer.fit(n_epochs)
