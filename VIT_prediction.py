import math
import click
from experiments.prediction import poseVIT_prediction
import wandb


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
    poseVIT_prediction(labels, dataset, experiment, dry_run)


if __name__ == "__main__":
    run_experiment()
