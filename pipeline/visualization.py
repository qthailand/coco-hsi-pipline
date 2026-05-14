from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ArrayLike = np.ndarray


def plot_mean_spectra_by_class(
    spectra: ArrayLike,
    labels: ArrayLike,
    wavelengths: Optional[ArrayLike] = None,
    max_classes: Optional[int] = None,
    with_std: bool = True,
    title: str = "Mean Spectra by Class",
    alpha_band: float = 0.2,
    figsize: Tuple[float, float] = (10.0, 6.0),
) -> plt.Figure:
    """Plot mean spectra per class with optional std band.

    Parameters
    ----------
    spectra : np.ndarray, shape (N, C)
        Input spectra.
    labels : np.ndarray, shape (N,)
        Class label per spectrum.
    wavelengths : np.ndarray, shape (C,), optional
        Wavelength axis. If None, band index is used.
    max_classes : int | None
        Plot only the first K classes if provided.
    with_std : bool
        Whether to draw mean +/- std as transparent area.
    """
    X = _validate_2d_array(spectra, name="spectra")
    y = _validate_1d_labels(labels, expected_len=X.shape[0], name="labels")

    if wavelengths is None:
        x_axis = np.arange(X.shape[1], dtype=np.float32)
        x_label = "Band Index"
    else:
        w = np.asarray(wavelengths, dtype=np.float32)
        if w.ndim != 1 or w.shape[0] != X.shape[1]:
            raise ValueError(
                f"wavelengths must be 1D with length {X.shape[1]}, got shape {w.shape}"
            )
        x_axis = w
        x_label = "Wavelength"

    classes = np.unique(y)
    if max_classes is not None:
        if max_classes < 1:
            raise ValueError("max_classes must be >= 1")
        classes = classes[:max_classes]

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab20")

    for i, cls in enumerate(classes):
        mask = y == cls
        Xc = X[mask]
        if Xc.shape[0] == 0:
            continue

        mean_curve = Xc.mean(axis=0)
        std_curve = Xc.std(axis=0)
        color = cmap(i % 20)

        ax.plot(x_axis, mean_curve, color=color, linewidth=1.8, label=f"Class {cls}")
        if with_std:
            lower = mean_curve - std_curve
            upper = mean_curve + std_curve
            ax.fill_between(x_axis, lower, upper, color=color, alpha=alpha_band)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Reflectance / Intensity")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: ArrayLike,
    labels: Optional[Sequence[object]] = None,
    normalize: bool = False,
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (7.0, 6.0),
    annotate: bool = True,
) -> plt.Figure:
    """Visualize confusion matrix as a heatmap."""
    matrix = np.asarray(cm)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cm must be a square 2D matrix")

    if labels is None:
        tick_labels = [str(i) for i in range(matrix.shape[0])]
    else:
        if len(labels) != matrix.shape[0]:
            raise ValueError("labels length must match confusion matrix size")
        tick_labels = [str(x) for x in labels]

    if normalize:
        denom = matrix.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        view = matrix / denom
        fmt = ".2f"
    else:
        view = matrix
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(view, interpolation="nearest", cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title + (" (Normalized)" if normalize else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    if annotate:
        threshold = view.max() / 2.0 if view.size else 0.0
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                value = view[i, j]
                if fmt == "d":
                    text = format(int(value), fmt)
                else:
                    text = format(float(value), fmt)
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    return fig


def plot_2d_feature_scatter(
    features_2d: ArrayLike,
    labels: ArrayLike,
    title: str = "2D Feature Scatter",
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
    alpha: float = 0.6,
    s: float = 12.0,
    figsize: Tuple[float, float] = (8.0, 6.0),
    max_points_per_class: Optional[int] = 5000,
    random_state: Optional[int] = 42,
) -> plt.Figure:
    """Scatter plot for 2D features, colored by class labels."""
    X = _validate_2d_array(features_2d, name="features_2d")
    if X.shape[1] != 2:
        raise ValueError(f"features_2d must have shape (N, 2), got {X.shape}")

    y = _validate_1d_labels(labels, expected_len=X.shape[0], name="labels")
    rng = np.random.default_rng(random_state)

    fig, ax = plt.subplots(figsize=figsize)
    classes = np.unique(y)
    cmap = plt.get_cmap("tab20")

    for i, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue

        if max_points_per_class is not None and max_points_per_class > 0 and idx.size > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)

        color = cmap(i % 20)
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            s=s,
            alpha=alpha,
            color=color,
            edgecolors="none",
            label=f"Class {cls} (n={idx.size})",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_class_distribution(
    labels: ArrayLike,
    title: str = "Class Distribution",
    figsize: Tuple[float, float] = (8.0, 4.8),
) -> plt.Figure:
    """Bar plot of class sample counts."""
    y = _validate_1d_labels(labels, expected_len=None, name="labels")
    classes, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(np.arange(classes.shape[0]), counts, color="#2a9d8f", alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(np.arange(classes.shape[0]))
    ax.set_xticklabels([str(c) for c in classes])
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to file."""
    if dpi < 50:
        raise ValueError("dpi should be >= 50")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def _validate_2d_array(X: ArrayLike, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (N, C)")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have non-zero shape")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or inf values")
    return X


def _validate_1d_labels(y: ArrayLike, expected_len: Optional[int], name: str) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if expected_len is not None and y.shape[0] != expected_len:
        raise ValueError(f"{name} length must be {expected_len}, got {y.shape[0]}")
    return y


__all__ = [
    "plot_mean_spectra_by_class",
    "plot_confusion_matrix",
    "plot_2d_feature_scatter",
    "plot_class_distribution",
    "save_figure",
]
