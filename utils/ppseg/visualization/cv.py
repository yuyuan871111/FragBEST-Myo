import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _cv_subplot_individual(ax, **kwargs):
    for i, data in enumerate(kwargs["data_list"]):
        ax.plot(data[kwargs["tag"]], label=f"Fold {i + 1}", linestyle="--")
    return ax


def _cv_subplot_average(ax, fill_between_color="grey", **kwargs):
    ax.plot(kwargs["data_cv_avg"][kwargs["tag"]], label="Avg.")
    data_upper = (
        kwargs["data_cv_avg"][kwargs["tag"]] + kwargs["data_cv_std"][kwargs["tag"]]
    )
    data_lower = (
        kwargs["data_cv_avg"][kwargs["tag"]] - kwargs["data_cv_std"][kwargs["tag"]]
    )
    ax.fill_between(
        kwargs["data_cv_avg"].index,
        data_lower,
        data_upper,
        color=fill_between_color,
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    return ax


def _cv_subplot(ax, **kwargs):
    ax = _cv_subplot_individual(ax, **kwargs)
    ax = _cv_subplot_average(ax, **kwargs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(kwargs["name"])
    ax.legend()
    return ax


def plot_training_process_cv(data_list, data_cv_avg, data_cv_std):
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), dpi=300)
    axes = axes.flatten()
    tags = [
        "training_accuracy",
        "validation_accuracy",
        "training_mIoU",
        "validation_mIoU",
    ]
    names = ["Train Accuracy", "Val Accuracy", "Train mIOU", "Val mIOU"]

    for ax, tag, name in zip(axes, tags, names):
        data = {
            "data_list": data_list,
            "data_cv_avg": data_cv_avg,
            "data_cv_std": data_cv_std,
            "tag": tag,
            "name": name,
        }
        ax = _cv_subplot(ax, **data)

    return fig, axes


def plot_training_process_cv_avg(data_cv_avg, data_cv_std):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), dpi=300)
    axes = axes.flatten()
    tags = [
        "training_accuracy",
        "validation_accuracy",
        "training_mIoU",
        "validation_mIoU",
    ]
    names = ["Train Accuracy", "Val Accuracy", "Train mIOU", "Val mIOU"]
    colors = ["tab:blue", "tab:orange", "tab:blue", "tab:orange"]

    for idx, (tag, name, color) in enumerate(zip(tags, names, colors)):
        data = {
            "data_list": [],
            "data_cv_avg": data_cv_avg,
            "data_cv_std": data_cv_std,
            "tag": tag,
            "name": name,
        }
        axes[idx // 2] = _cv_subplot_average(
            axes[idx // 2], fill_between_color=color, **data
        )

    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("mIOU")
    axes[0].legend(["Train (Avg.)", "Train (std.)", "Val (Avg.)", "Val (std.)"])
    axes[1].legend(["Train (Avg.)", "Train (std.)", "Val (Avg.)", "Val (std.)"])

    return fig, axes


def plot_cv_indices(cv, ax, X, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    cmap_cv = plt.cm.coolwarm

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    yticklabels = list(range(1, n_splits + 1))
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits, -0.2],
        xlim=[0, 2700],
    )
    ax.set_title(f"{type(cv).__name__}", fontsize=15)
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Validation set", "Training set"],
        loc=(1.02, 0.8),
    )
    return ax
