from typing import NewType, TypeVar, Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from metrics.utils import print_metrics

DataLoaders = NewType("DataLoaders", dict[Literal["train", "test"], DataLoader])
Model = TypeVar("Model", bound=torch.nn.Module)
Optimizer = TypeVar("Optimizer", bound=torch.optim.Optimizer)
Criterion = TypeVar("Criterion", bound=torch.nn.Module)


class ClassificationTrainer:
    def __init__(
        self,
        data: DataLoaders,
        model: Model,
        optimizer: Optimizer,
        criterion: Criterion,
        scheduler=None,
        device: torch.device = None,
        verbose: bool = True,
        gradient_clip=False,
    ):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.verbose = verbose
        self.gradient_clip = gradient_clip

        self.train_metrics = []
        self.test_metrics = []

        self.current_epoch = 0

    def add_train_metric(self, name, metric):
        self.train_metrics.append((name, metric))

    def add_test_metric(self, name, metric):
        self.test_metrics.append((name, metric))

    def make_prediction(self, features, targets):
        logits = self.model(features)
        prediction = torch.max(logits, -1)[1]

        loss = self.criterion(logits, targets)

        return logits, prediction, loss

    def train_epoch(self):
        self.model.train()

        for _, metric in self.train_metrics:
            metric.reset()

        progress_bar = tqdm(
            self.data["train"], desc="Training", disable=(not self.verbose)
        )

        for index, (features, target) in enumerate(progress_bar):
            features = features.to(self.device)
            target = target.to(self.device)

            logits, prediction, loss = self.make_prediction(features, target)

            # Compute metrics
            for _, metric in self.train_metrics:
                metric(logits, target)

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            self.optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        print_metrics(self.train_metrics)

    def test_epoch(self):
        self.model.eval()

        for _, metric in self.test_metrics:
            metric.reset()

        progress_bar = tqdm(
            self.data["test"], desc="Testing", disable=(not self.verbose)
        )

        for index, (features, target) in enumerate(progress_bar):
            features = features.to(self.device)
            target = target.to(self.device)

            logits, prediction, loss = self.make_prediction(features, target)

            # Compute metrics
            for _, metric in self.test_metrics:
                metric(logits, target)

            progress_bar.set_postfix(loss=loss.item())

        print_metrics(self.test_metrics)

    def fit(self, epochs: int, save_best: bool = False):
        start = datetime.now()
        self.model.to(self.device)
        best_metric = None

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            print("-" * 10, "EPOCH", epoch + 1, "/", epochs)
            self.train_epoch()
            self.test_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            # SAVE EXPERIMENT AND MODEL

        delta = datetime.now() - start
        print("-" * 10)
        print(f"TRAINING COMPLETED [{delta}].")
