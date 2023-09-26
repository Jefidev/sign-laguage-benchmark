import wandb


class WandbLogger:
    def __init__(self, project, config, model=None, criterion=None) -> None:
        wandb.init(project, config=config)

        if model != None:
            assert criterion != None, "Criterion must be provided if model is provided"
            wandb.watch(model, criterion=criterion)

    def log_metrics(self, train_metrics, test_metrics):
        for name, metric in train_metrics:
            wandb.log({"Train " + name: metric})

        for name, metric in test_metrics:
            wandb.log({"Valid" + name: metric})

    def log_losses(self, train_loss, test_loss):
        wandb.log({"Train Loss": train_loss})
        wandb.log({"Valid Loss": test_loss})
