from models import PoseViT
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from schedulers import WarmupLinearScheduler
import math
from training import ClassificationTrainer
from utils import load_datasets, set_common_metrics
import wandb


def poseVIT_prediction(n_labels, dataset_path, project_name, dry_run):
    # Default configuration
    config_defaults = {
        "n_labels": 250,
        "seq_size": 32,
        "n_epochs": 300,
        "data_augmentation": False,
        "gradient_clip": False,
        "batch_size": 128,
        "embedding_size": 64,
        "dataset": "/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/isol",
        "dry_run": True,
    }

    # Sweep configuration
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "valid balanced accuracy", "goal": "maximize"},
        "parameters": {
            "seq_size": {"values": [16, 32, 64]},
            "data_augmentation": {"values": [True, False]},
            "gradient_clip": {"values": [True, False]},
            "batch_size": {"values": [128, 256, 512]},
            "embedding_size": {"values": [32, 64, 128, 256]},
        },
    }

    config_defaults["n_labels"] = n_labels
    config_defaults["dataset"] = dataset_path
    config_defaults["dry_run"] = dry_run

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    # run sweep
    wandb.agent(sweep_id, function=start_run, count=25)
    wandb.init(config=config_defaults)


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
