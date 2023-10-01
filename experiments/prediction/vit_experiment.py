from models import PoseViT
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from schedulers import WarmupLinearScheduler
import math
from training import ClassificationTrainer
from utils import load_datasets, set_common_metrics
import wandb


def start_run():
    config = wandb.config
    warmups_steps = math.floor(config["n_epochs"] * 0.2)

    # Load the datasets
    train_dataset, test_dataset = load_datasets(
        config["dataset"],
        config["seq_size"],
        config["n_labels"],
        config["data_augmentation"],
        config["dry_run"],
    )

    # Create data loader
    data = {}

    data["train"] = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8
    )
    data["test"] = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8
    )

    # Create model
    model = PoseViT(
        config["n_labels"],
        embedding_size=config["embedding_size"],
        seq_size=config["seq_size"],
    )

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Scheduler
    scheduler = WarmupLinearScheduler(optimizer, warmups_steps, config["n_epochs"])

    # Create trainer
    trainer = ClassificationTrainer(
        data,
        model,
        optimizer,
        criterion,
        scheduler,
        gradient_clip=config["gradient_clip"],
    )

    # Attach metrics
    set_common_metrics(trainer, config["n_labels"])

    # Run training
    trainer.fit(config["n_epochs"])
