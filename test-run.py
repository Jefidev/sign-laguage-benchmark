import math
import click
from experiments.prediction import poseVIT_prediction
import wandb

config_defaults = {
    "n_labels": 250,
    "seq_size": 32,
    "n_epochs": 10,
    "data_augmentation": False,
    "gradient_clip": False,
    "batch_size": 128,
    "embedding_size": 64,
    "dataset": "/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/isol",
    "dry_run": True,
}


@click.command()
@click.option("-l", "--labels", default=250, help="Number of labels to predict")
@click.option(
    "-e",
    "--experiment",
    default="test-run",
    help="Name of the Experiment to run",
)
@click.option(
    "-d",
    "--dataset",
    default="/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/isol",
    help="Path to the LSFB dataset",
)
@click.option("--dry-run", is_flag=True)
def run_experiment(labels, experiment, dataset, dry_run):
    """Run Sign Language Prediction Experiment"""

    config_defaults["n_labels"] = labels
    config_defaults["dataset"] = dataset
    config_defaults["dry_run"] = dry_run
    wandb.init(config=config_defaults, project=experiment)
    poseVIT_prediction()


if __name__ == "__main__":
    run_experiment()
