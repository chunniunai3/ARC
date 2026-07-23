import matplotlib.pyplot as plt
import numpy as np


def plot_grid(grid: np.ndarray, ax: plt.Axes | None = None, title: str = "") -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid, cmap="tab10", vmin=0, vmax=9)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="black", linewidth=0.5)
    if title:
        ax.set_title(title)
    return ax


def show_task(task: dict, max_examples: int = 3) -> None:
    n_train = min(len(task["train"]), max_examples)
    n_test = min(len(task["test"]), max_examples)
    total = n_train + n_test
    fig, axes = plt.subplots(2, total, figsize=(4 * total, 4))

    for i in range(n_train):
        pair = task["train"][i]
        plot_grid(np.array(pair["input"]), axes[0, i], f"Train {i+1} Input")
        plot_grid(np.array(pair["output"]), axes[1, i], f"Train {i+1} Output")

    for i in range(n_test):
        pair = task["test"][i]
        col = n_train + i
        plot_grid(np.array(pair["input"]), axes[0, col], f"Test {i+1} Input")
        if pair.get("output"):
            plot_grid(np.array(pair["output"]), axes[1, col], f"Test {i+1} Output")

    plt.tight_layout()
    plt.show()
