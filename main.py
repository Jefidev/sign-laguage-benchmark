import math
import click
from experiments.prediction import poseVIT_prediction
from loggers.wandb_logger import WandbLogger
import wandb


@click.command()
@click.option("-e", "--experiment", help="Name of the experiment to run", required=True)
@click.option(
    "-d",
    "--dataset",
    default="/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/isol",
    help="Path to the LSFB dataset",
)
@click.option("--dry-run", is_flag=True)
def run_experiment(experiment, dataset, dry_run):
    """Run Sign Language Prediction Experiment"""

    if experiment == "posevit-prediction":
        # Sweep configuration
        sweep_config = {
            "method": "grid",
            "metric": {"name": "Valid balanced accuracy", "goal": "maximize"},
            "parameters": {
                "n_labels": {"values": [250, 500, 1000, 2000]},
                "seq_size": {"values": [32, 64, 128, 256]},
                "n_epochs": {"values": [10, 100, 1000]},
                "data_augmentation": {"values": [True, False]},
                "gradient_clip": {"values": [True, False]},
                "batch_size": {"values": [32, 64, 128, 256, 512]},
                "embedding_size": {"values": [32, 64, 128, 256]},
            },
        }

        config_defaults = {
            "n_labels": 250,
            "seq_size": 32,
            "n_epochs": 10,
            "data_augmentation": False,
            "gradient_clip": False,
            "batch_size": 128,
            "embedding_size": 64,
            "dataset": dataset,
            "dry_run": dry_run,
        }

        sweep_id = wandb.sweep(sweep_config, project="test_sweep")
        wandb.init(config=config_defaults)

        # run sweep
        wandb.agent(sweep_id, function=poseVIT_prediction)


if __name__ == "__main__":
    run_experiment()
