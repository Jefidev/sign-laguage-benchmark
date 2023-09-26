from tabulate import tabulate
from metrics import Metric


def print_metrics(metrics: list[Metric]):
    table_content = [[], []]
    for name, metric in metrics:
        value = metric.compute()

        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                table_content[0].append(f"{name} {i}")
                table_content[1].append(f"{v:.4f}")
        else:
            table_content[0].append(name)
            table_content[1].append(f"{value:.4f}")
    print(tabulate(table_content, headers="firstrow", tablefmt="fancy_grid"))
