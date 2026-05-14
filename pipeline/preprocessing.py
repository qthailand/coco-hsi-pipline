from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray


def encode_str_labels(raw_labels: ArrayLike) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """Encode string category names to consecutive integers (1-based, alphabetical)."""
    raw = np.asarray(raw_labels)
    if raw.ndim != 1:
        raise ValueError("raw_labels must be a 1D array")

    unique_names = sorted(set(raw.tolist()))
    name_to_id = {name: idx + 1 for idx, name in enumerate(unique_names)}
    id_to_name = {v: k for k, v in name_to_id.items()}
    labels_int = np.array([name_to_id[n] for n in raw], dtype=np.int32)
    label_order = np.array(sorted(id_to_name.keys()), dtype=np.int32)
    return labels_int, id_to_name, label_order


def samplewise_dev_test_split(
    labels: ArrayLike,
    source_files: ArrayLike,
    test_ratio: float = 0.2,
    random_state: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split pixels by class-specific unique samples (source_file IDs)."""
    labels_arr = np.asarray(labels, dtype=np.int32)
    source_arr = np.asarray(source_files, dtype=str)

    if labels_arr.ndim != 1 or source_arr.ndim != 1:
        raise ValueError("labels and source_files must be 1D arrays")
    if labels_arr.shape[0] != source_arr.shape[0]:
        raise ValueError("labels and source_files must have the same length")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1")

    unique_files = np.unique(source_arr)
    if unique_files.size < 2:
        raise ValueError(
            "Dataset must contain at least 2 unique source files for sample-wise split."
        )

    rng = np.random.default_rng(random_state)
    insufficient: list[tuple[int, int]] = []
    labels_by_file: dict[str, set[int]] = {}

    for class_id in np.unique(labels_arr):
        class_mask = labels_arr == class_id
        unique_samples = np.unique(source_arr[class_mask])
        if unique_samples.size < 2:
            insufficient.append((int(class_id), int(unique_samples.size)))
            continue
        for src in unique_samples.tolist():
            labels_by_file.setdefault(src, set()).add(int(class_id))

    if insufficient:
        err_parts = [f"class {c} has {n} sample(s)" for c, n in insufficient]
        raise ValueError(
            "Each class must have at least 2 distinct pixel-containing samples "
            "for sample-wise train/test split. "
            + "; ".join(err_parts)
        )

    n_test_files = int(np.floor(unique_files.size * test_ratio))
    n_test_files = max(1, min(n_test_files, unique_files.size - 1))

    shuffled = unique_files.copy()
    for _ in range(200):
        rng.shuffle(shuffled)
        test_files = set(shuffled[:n_test_files].tolist())
        dev_files = set(shuffled[n_test_files:].tolist())

        valid = True
        for class_id in np.unique(labels_arr):
            class_sources = {
                src for src, labels_set in labels_by_file.items()
                if class_id in labels_set
            }
            if not class_sources:
                valid = False
                break
            if not (class_sources & test_files):
                valid = False
                break
            if not (class_sources & dev_files):
                valid = False
                break

        if valid:
            test_mask = np.isin(source_arr, np.asarray(list(test_files), dtype=str))
            dev_mask = ~test_mask
            dev_idx = np.flatnonzero(dev_mask)
            test_idx = np.flatnonzero(test_mask)
            if dev_idx.size > 0 and test_idx.size > 0:
                return dev_idx, test_idx

    raise ValueError(
        "Sample-wise split failed to find a valid dev/test partition after "
        f"200 attempts using test_ratio={test_ratio}."
    )


@dataclass
class PreprocessConfig:
    """Configuration for hyperspectral 2D preprocessing.

    Parameters
    ----------
    remove_wavelength_ranges : list[tuple[float, float]]
        Inclusive wavelength ranges to remove, for example noisy water-absorption regions.
    spectral_normalization : str
        Per-spectrum normalization method: one of {"none", "snv", "l2", "max", "area"}.
    global_scaling : str
        Global scaling method fit on train data and reused at inference:
        one of {"none", "standard", "minmax"}.
    clip_percentile : tuple[float, float] | None
        If set, clip each band using percentiles computed from train data.
    eps : float
        Numeric stability constant used for safe division.
    """

    remove_wavelength_ranges: List[Tuple[float, float]] = field(default_factory=list)
    spectral_normalization: str = "snv"
    global_scaling: str = "standard"
    clip_percentile: Optional[Tuple[float, float]] = None
    eps: float = 1e-8


class HSIPreprocessor:
    """Train/inference-safe preprocessor for HSI2D arrays.

    Input arrays are expected in shape (N_pixel, N_band).
    """

    _VALID_SPECTRAL = {"none", "snv", "l2", "max", "area"}
    _VALID_GLOBAL = {"none", "standard", "minmax"}

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

        if self.config.spectral_normalization not in self._VALID_SPECTRAL:
            raise ValueError(
                f"spectral_normalization must be one of {sorted(self._VALID_SPECTRAL)}"
            )
        if self.config.global_scaling not in self._VALID_GLOBAL:
            raise ValueError(f"global_scaling must be one of {sorted(self._VALID_GLOBAL)}")

        self._is_fitted: bool = False
        self._band_mask: Optional[np.ndarray] = None
        self._band_clip_low: Optional[np.ndarray] = None
        self._band_clip_high: Optional[np.ndarray] = None
        self._scale_a: Optional[np.ndarray] = None
        self._scale_b: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, wavelengths: Optional[ArrayLike] = None) -> "HSIPreprocessor":
        """Fit preprocessing statistics from train spectra only."""
        X = _validate_2d_array(X, name="X")

        band_mask = _build_band_mask(
            n_bands=X.shape[1],
            wavelengths=wavelengths,
            remove_ranges=self.config.remove_wavelength_ranges,
        )
        X_masked = X[:, band_mask]

        if self.config.clip_percentile is not None:
            low, high = self.config.clip_percentile
            if not (0.0 <= low < high <= 100.0):
                raise ValueError("clip_percentile must be in [0, 100] and low < high")
            self._band_clip_low = np.percentile(X_masked, low, axis=0)
            self._band_clip_high = np.percentile(X_masked, high, axis=0)
        else:
            self._band_clip_low = None
            self._band_clip_high = None

        X_fit = self._apply_clip(X_masked)
        X_fit = _apply_spectral_normalization(
            X_fit,
            method=self.config.spectral_normalization,
            eps=self.config.eps,
        )

        if self.config.global_scaling == "standard":
            mu = X_fit.mean(axis=0)
            sigma = X_fit.std(axis=0)
            sigma = np.where(sigma < self.config.eps, 1.0, sigma)
            self._scale_a = mu
            self._scale_b = sigma
        elif self.config.global_scaling == "minmax":
            x_min = X_fit.min(axis=0)
            x_max = X_fit.max(axis=0)
            span = x_max - x_min
            span = np.where(span < self.config.eps, 1.0, span)
            self._scale_a = x_min
            self._scale_b = span
        else:
            self._scale_a = None
            self._scale_b = None

        self._band_mask = band_mask
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Apply fitted preprocessing pipeline to any input spectra."""
        if not self._is_fitted:
            raise RuntimeError("HSIPreprocessor is not fitted. Call fit() first.")

        X = _validate_2d_array(X, name="X")
        if self._band_mask is None:
            raise RuntimeError("Internal state error: missing band mask.")
        if X.shape[1] != self._band_mask.shape[0]:
            raise ValueError(
                "Input band count does not match fitted preprocessor: "
                f"got {X.shape[1]}, expected {self._band_mask.shape[0]}"
            )

        X_out = X[:, self._band_mask]
        X_out = self._apply_clip(X_out)
        X_out = _apply_spectral_normalization(
            X_out,
            method=self.config.spectral_normalization,
            eps=self.config.eps,
        )
        X_out = self._apply_global_scaling(X_out)
        return X_out

    def fit_transform(self, X: ArrayLike, wavelengths: Optional[ArrayLike] = None) -> ArrayLike:
        """Fit and transform train spectra in one call."""
        return self.fit(X, wavelengths=wavelengths).transform(X)

    def transform_wavelengths(self, wavelengths: ArrayLike) -> ArrayLike:
        """Apply fitted band mask to wavelength vector."""
        if not self._is_fitted or self._band_mask is None:
            raise RuntimeError("HSIPreprocessor is not fitted. Call fit() first.")
        wavelengths = np.asarray(wavelengths, dtype=float)
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1D array")
        if wavelengths.shape[0] != self._band_mask.shape[0]:
            raise ValueError(
                "Wavelength count does not match fitted preprocessor: "
                f"got {wavelengths.shape[0]}, expected {self._band_mask.shape[0]}"
            )
        return wavelengths[self._band_mask]

    @property
    def band_mask_(self) -> np.ndarray:
        """Boolean mask of kept bands after fitting."""
        if not self._is_fitted or self._band_mask is None:
            raise RuntimeError("HSIPreprocessor is not fitted. Call fit() first.")
        return self._band_mask.copy()

    def _apply_clip(self, X: ArrayLike) -> ArrayLike:
        if self._band_clip_low is None or self._band_clip_high is None:
            return X
        return np.clip(X, self._band_clip_low, self._band_clip_high)

    def _apply_global_scaling(self, X: ArrayLike) -> ArrayLike:
        if self.config.global_scaling == "none":
            return X

        if self._scale_a is None or self._scale_b is None:
            raise RuntimeError("Internal state error: missing global scaling statistics.")

        if self.config.global_scaling == "standard":
            return (X - self._scale_a) / self._scale_b
        if self.config.global_scaling == "minmax":
            return (X - self._scale_a) / self._scale_b

        raise RuntimeError("Unsupported global scaling configuration.")


def stratified_train_val_split(
    labels: ArrayLike,
    val_ratio: float = 0.2,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stratified train/val indices from label vector.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Class labels for each sample.
    val_ratio : float, default=0.2
        Ratio of validation samples in each class.
    random_state : int | None
        Seed for reproducible split.
    """

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    train_idx_parts: List[np.ndarray] = []
    val_idx_parts: List[np.ndarray] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        n_val = int(np.floor(len(cls_idx) * val_ratio))
        if n_val == 0 and len(cls_idx) > 1:
            n_val = 1
        if n_val >= len(cls_idx):
            n_val = len(cls_idx) - 1

        val_idx_parts.append(cls_idx[:n_val])
        train_idx_parts.append(cls_idx[n_val:])

    train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _validate_2d_array(X: ArrayLike, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (N, C)")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have non-zero shape")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or inf values")
    return X


def _build_band_mask(
    n_bands: int,
    wavelengths: Optional[ArrayLike],
    remove_ranges: Sequence[Tuple[float, float]],
) -> np.ndarray:
    band_mask = np.ones(n_bands, dtype=bool)
    if not remove_ranges:
        return band_mask

    if wavelengths is None:
        raise ValueError(
            "remove_wavelength_ranges was set, but wavelengths is None. "
            "Provide wavelength vector from HSI loader."
        )

    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1:
        raise ValueError("wavelengths must be 1D")
    if wl.shape[0] != n_bands:
        raise ValueError(
            f"wavelength length ({wl.shape[0]}) must match band count ({n_bands})"
        )

    for low, high in remove_ranges:
        if low > high:
            low, high = high, low
        band_mask &= ~((wl >= low) & (wl <= high))

    if not np.any(band_mask):
        raise ValueError("All bands were removed by remove_wavelength_ranges")

    return band_mask


def _apply_spectral_normalization(X: ArrayLike, method: str, eps: float) -> ArrayLike:
    if method == "none":
        return X

    if method == "snv":
        mu = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True)
        sigma = np.where(sigma < eps, 1.0, sigma)
        return (X - mu) / sigma

    if method == "l2":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm = np.where(norm < eps, 1.0, norm)
        return X / norm

    if method == "max":
        vmax = np.max(np.abs(X), axis=1, keepdims=True)
        vmax = np.where(vmax < eps, 1.0, vmax)
        return X / vmax

    if method == "area":
        area = np.sum(np.abs(X), axis=1, keepdims=True)
        area = np.where(area < eps, 1.0, area)
        return X / area

    raise ValueError(f"Unsupported spectral normalization method: {method}")


__all__ = [
    "PreprocessConfig",
    "HSIPreprocessor",
    "stratified_train_val_split",
]
