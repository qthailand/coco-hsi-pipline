from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass
class FeatureExtractionConfig:
    """Configuration for HSI feature extraction.

    Parameters
    ----------
    mode : str
        Feature mode, one of:
        - "raw": keep input spectra as features
        - "stat": hand-crafted statistics from each spectrum
        - "pca": PCA embedding from spectra
        - "pca_stat": concatenate PCA and stat features
    pca_components : int | float | None
        If int: number of PCA components.
        If float in (0, 1]: keep enough components to reach explained variance ratio.
        If None: PCA is skipped unless mode requires PCA, then all valid components are used.
    center : bool
        Whether to center features before PCA.
    scale : bool
        Whether to standardize features before PCA.
    eps : float
        Numeric stability term used for safe division.
    """

    mode: str = "pca_stat"
    pca_components: Optional[float] = 0.99
    center: bool = True
    scale: bool = False
    eps: float = 1e-8


class HSIFeatureExtractor:
    """Train/inference-safe feature extractor for 2D HSI spectra.

    Expected input shape is (N_pixel, N_band).
    """

    _VALID_MODES = {"raw", "stat", "pca", "pca_stat"}

    def __init__(self, config: Optional[FeatureExtractionConfig] = None) -> None:
        self.config = config or FeatureExtractionConfig()
        if self.config.mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}")

        self._is_fitted: bool = False
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._pca_evr: Optional[np.ndarray] = None
        self._raw_dim: Optional[int] = None

    def fit(self, X: ArrayLike) -> "HSIFeatureExtractor":
        """Fit extraction statistics from train spectra."""
        X = _validate_2d_array(X, name="X")
        self._raw_dim = X.shape[1]

        X_pca_in = self._prepare_for_pca(X)
        if self.config.mode in {"pca", "pca_stat"}:
            self._fit_pca(X_pca_in)

        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Extract features from spectra using fitted parameters."""
        if not self._is_fitted:
            raise RuntimeError("HSIFeatureExtractor is not fitted. Call fit() first.")

        X = _validate_2d_array(X, name="X")
        if self._raw_dim is None:
            raise RuntimeError("Internal state error: missing fitted input dimension.")
        if X.shape[1] != self._raw_dim:
            raise ValueError(
                "Input band count does not match fitted extractor: "
                f"got {X.shape[1]}, expected {self._raw_dim}"
            )

        parts = []
        if self.config.mode in {"raw"}:
            parts.append(X.astype(np.float32, copy=False))
        if self.config.mode in {"stat", "pca_stat"}:
            parts.append(_extract_stat_features(X, eps=self.config.eps))
        if self.config.mode in {"pca", "pca_stat"}:
            X_pca_in = self._apply_pca_input_transform(X)
            parts.append(self._project_pca(X_pca_in))

        if not parts:
            raise RuntimeError("No feature parts were produced. Check extractor mode.")
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """Fit and transform train spectra in one call."""
        return self.fit(X).transform(X)

    @property
    def output_dim_(self) -> int:
        """Return output feature dimension after fitting."""
        if not self._is_fitted:
            raise RuntimeError("HSIFeatureExtractor is not fitted. Call fit() first.")
        raw_dim = self._raw_dim or 0
        stat_dim = 10
        pca_dim = 0 if self._pca_components is None else self._pca_components.shape[1]

        if self.config.mode == "raw":
            return raw_dim
        if self.config.mode == "stat":
            return stat_dim
        if self.config.mode == "pca":
            return pca_dim
        return stat_dim + pca_dim

    @property
    def pca_explained_variance_ratio_(self) -> np.ndarray:
        """Explained variance ratio per PCA component."""
        if self._pca_evr is None:
            raise RuntimeError("PCA is not fitted in current mode.")
        return self._pca_evr.copy()

    def _prepare_for_pca(self, X: np.ndarray) -> np.ndarray:
        if self.config.center:
            mu = X.mean(axis=0)
            self._mu = mu
            X_out = X - mu
        else:
            self._mu = None
            X_out = X

        if self.config.scale:
            sigma = X_out.std(axis=0)
            sigma = np.where(sigma < self.config.eps, 1.0, sigma)
            self._sigma = sigma
            X_out = X_out / sigma
        else:
            self._sigma = None

        return X_out

    def _apply_pca_input_transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X
        if self._mu is not None:
            X_out = X_out - self._mu
        if self._sigma is not None:
            X_out = X_out / self._sigma
        return X_out

    def _fit_pca(self, X: np.ndarray) -> None:
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to fit PCA")

        _, svals, vt = np.linalg.svd(X, full_matrices=False)
        components_all = vt.T

        denom = max(n_samples - 1, 1)
        explained_var = (svals ** 2) / denom
        total_var = explained_var.sum()
        if total_var <= self.config.eps:
            evr = np.zeros_like(explained_var)
        else:
            evr = explained_var / total_var

        n_keep = _resolve_num_components(
            pca_components=self.config.pca_components,
            max_components=min(n_samples, n_features),
            explained_variance_ratio=evr,
        )

        self._pca_components = components_all[:, :n_keep]
        self._pca_evr = evr[:n_keep]

    def _project_pca(self, X: np.ndarray) -> np.ndarray:
        if self._pca_components is None:
            raise RuntimeError("PCA is not fitted in current mode.")
        return X @ self._pca_components


def _resolve_num_components(
    pca_components: Optional[float],
    max_components: int,
    explained_variance_ratio: np.ndarray,
) -> int:
    if max_components < 1:
        raise ValueError("max_components must be >= 1")

    if pca_components is None:
        return max_components

    if isinstance(pca_components, (int, np.integer)):
        n_keep = int(pca_components)
        if n_keep < 1:
            raise ValueError("pca_components must be >= 1 when integer")
        return min(n_keep, max_components)

    if isinstance(pca_components, (float, np.floating)):
        ratio = float(pca_components)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("pca_components as float must be in (0, 1]")
        cum = np.cumsum(explained_variance_ratio)
        n_keep = int(np.searchsorted(cum, ratio, side="left") + 1)
        return min(max(n_keep, 1), max_components)

    raise TypeError("pca_components must be int, float, or None")


def _extract_stat_features(X: np.ndarray, eps: float) -> np.ndarray:
    # Simple spectral descriptors that work well as compact baseline features.
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    vmin = X.min(axis=1, keepdims=True)
    vmax = X.max(axis=1, keepdims=True)
    ptp = vmax - vmin

    abs_x = np.abs(X)
    area = abs_x.sum(axis=1, keepdims=True)
    l2 = np.linalg.norm(X, axis=1, keepdims=True)
    mean_abs = abs_x.mean(axis=1, keepdims=True)

    # Spectral slope between start and end bands.
    slope = (X[:, -1:] - X[:, :1]) / max(X.shape[1] - 1, 1)

    std_safe = np.where(std < eps, 1.0, std)
    skew_like = (((X - mean) / std_safe) ** 3).mean(axis=1, keepdims=True)

    feats = np.concatenate(
        [mean, std, vmin, vmax, ptp, area, l2, mean_abs, slope, skew_like],
        axis=1,
    )
    return feats.astype(np.float32, copy=False)


def _validate_2d_array(X: ArrayLike, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (N, C)")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have non-zero shape")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or inf values")
    return X


__all__ = [
    "FeatureExtractionConfig",
    "HSIFeatureExtractor",
]
