from models import PoseViT
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pytorch_metric_learning import losses
from schedulers import WarmupLinearScheduler
import math
from training import ContrastiveTrainer
from utils import load_datasets, set_common_metrics
import wandb


def poseVIT_contrastive(n_labels, dataset_path, project_name, dry_run):
    # Default configuration
    config_defaults = {
        "n_labels": n_labels,
        "seq_size": 32,
        "n_epochs": 500,
        "data_augmentation": True,
        "gradient_clip": False,
        "batch_size": 128,
        "hidden_size": 64,
        "embedding_size": 64,
        "criterion": "NTXent",
        "dataset": dataset_path,
        "dry_run": dry_run,
    }

    # Sweep configuration

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "valid balanced accuracy", "goal": "maximize"},
        "parameters": {
            "seq_size": {"values": [16, 32, 64]},
            "batch_size": {"values": [256, 512, 1024]},
            "hidden_size": {"values": [32, 64, 128, 256]},
            "embedding_size": {"values": [32, 64, 128, 256]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    #  run sweep
    wandb.init(config=config_defaults)
    wandb.agent(sweep_id, function=start_run, count=25)


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
        embedding_size=config["hidden_size"],
        seq_size=config["seq_size"],
        contrastive=True,
        contrastive_embedding=config["embedding_size"],
    )

    # Criterion

    if config["criterion"] == "triplet":
        criterion = losses.TripletMarginLoss()
    elif config["criterion"] == "NTXent":
        criterion = losses.NTXentLoss()
    else:
        raise ValueError("Criterion not supported")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Scheduler
    scheduler = WarmupLinearScheduler(optimizer, warmups_steps, config["n_epochs"])

    # Create trainer
    trainer = ContrastiveTrainer(
        data,
        model,
        optimizer,
        criterion,
        config["n_labels"],
        scheduler,
        gradient_clip=config["gradient_clip"],
    )

    # Attach metrics
    set_common_metrics(trainer, config["n_labels"])

    # Run training
    trainer.fit(config["n_epochs"])
