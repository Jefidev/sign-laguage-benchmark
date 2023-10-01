from typing import NewType, TypeVar, Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from metrics.utils import print_metrics
from utils import log_metrics, log_losses
import wandb
from classifiers.knn_classifier import knn_eval

DataLoaders = NewType("DataLoaders", dict[Literal["train", "test"], DataLoader])
Model = TypeVar("Model", bound=torch.nn.Module)
Optimizer = TypeVar("Optimizer", bound=torch.optim.Optimizer)
Criterion = TypeVar("Criterion", bound=torch.nn.Module)


class ContrastiveTrainer:
    def __init__(
        self,
        data: DataLoaders,
        model: Model,
        optimizer: Optimizer,
        criterion: Criterion,
        n_class: int,
        scheduler=None,
        device: torch.device = None,
        verbose: bool = True,
        gradient_clip=False,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device:", device)

        self.data = data
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.scheduler = scheduler if scheduler is not None else None

        self.device = device
        self.verbose = verbose
        self.gradient_clip = gradient_clip

        self.train_metrics = []
        self.test_metrics = []
        self.n_class = n_class

        self.current_epoch = 0

    def add_train_metric(self, name, metric):
        self.train_metrics.append((name, metric))

    def add_test_metric(self, name, metric):
        self.test_metrics.append((name, metric))

    def make_prediction(self, features, targets):
        embeddings = self.model(features)
        loss = self.criterion(embeddings, targets)

        return embeddings, loss

    def train_epoch(self):
        self.model.train()

        for _, metric in self.train_metrics:
            metric.reset()

        progress_bar = tqdm(
            self.data["train"], desc="Training", disable=(not self.verbose)
        )

        # Accumulator for loss
        loss_accum = 0

        # Accumulator embeddings
        embeddings_accum = None
        targets = None

        for index, (features, target) in enumerate(progress_bar):
            features = features.to(self.device)
            target = target.to(self.device)

            embeddings, loss = self.make_prediction(features, target)

            loss_accum += loss.item()

            # Accumulate input and output data
            if embeddings_accum is None:
                embeddings_accum = embeddings.cpu().detach()
                targets = target.cpu().detach()
            else:
                embeddings_accum = torch.cat(
                    (embeddings_accum, embeddings.cpu().detach())
                )
                targets = torch.cat((targets, target.cpu().detach()))

            # Compute metrics
            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            self.optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        probas, labels = knn_eval(
            embeddings_accum.numpy(), targets.numpy(), n_classes=self.n_class
        )

        # transforming to torch
        probas = torch.from_numpy(probas).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)

        # Compute metrics
        for _, metric in self.train_metrics:
            metric(probas, labels)

        print_metrics(self.train_metrics)

        return loss_accum / len(self.data["train"])

    def test_epoch(self):
        self.model.eval()

        for _, metric in self.test_metrics:
            metric.reset()

        progress_bar = tqdm(
            self.data["test"], desc="Testing", disable=(not self.verbose)
        )

        # accumulate loss
        loss_accum = 0

        # accumulate embeddings
        embeddings_accum = None
        targets = None

        for index, (features, target) in enumerate(progress_bar):
            features = features.to(self.device)
            target = target.to(self.device)

            embeddings, loss = self.make_prediction(features, target)
            loss_accum += loss.item()

            # Accumulate input and output data
            if embeddings_accum is None:
                embeddings_accum = embeddings.cpu().detach()
                targets = target.cpu().detach()
            else:
                embeddings_accum = torch.cat(
                    (embeddings_accum, embeddings.cpu().detach())
                )
                targets = torch.cat((targets, target.cpu().detach()))

            progress_bar.set_postfix(loss=loss.item())

        # Compute metrics
        preds, labels = knn_eval(
            embeddings_accum.numpy(), targets.numpy(), n_classes=self.n_class
        )

        # transforme to torch
        preds = torch.from_numpy(preds).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)

        for _, metric in self.test_metrics:
            metric(preds, labels)

        print_metrics(self.test_metrics)
        return loss_accum / len(self.data["test"])

    def fit(self, epochs: int, save_best: bool = False):
        start = datetime.now()
        self.model.to(self.device)
        best_metric = None

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            print("-" * 10, "EPOCH", epoch + 1, "/", epochs)
            t_loss = self.train_epoch()
            v_loss = self.test_epoch()

            if wandb.run is not None:
                print("Logging metrics...")
                log_metrics(self.train_metrics, self.test_metrics)
                log_losses(t_loss, v_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            # SAVE EXPERIMENT AND MODEL

        delta = datetime.now() - start
        print("-" * 10)
        print(f"TRAINING COMPLETED [{delta}].")
