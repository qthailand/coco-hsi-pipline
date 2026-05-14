from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class ModelConfig:
    """Configuration for HSI classifier training.

    Parameters
    ----------
    algorithm : str
        Classifier name: "nearest_centroid" or "knn".
    k_neighbors : int
        Number of neighbors for kNN (used when algorithm="knn").
    distance : str
        Distance metric: "euclidean" or "cosine".
    batch_size : int
        Batch size used during prediction to control memory usage.
    eps : float
        Numeric stability constant.
    """

    algorithm: str = "nearest_centroid"
    k_neighbors: int = 5
    distance: str = "euclidean"
    batch_size: int = 2048
    eps: float = 1e-8


class HSIModelTrainer:
    """Train/evaluate/predict utility for 2D HSI feature arrays.

    Expected feature shape is (N_sample, N_feature), labels shape is (N_sample,).
    """

    _VALID_ALGO = {"nearest_centroid", "knn"}
    _VALID_DISTANCE = {"euclidean", "cosine"}

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()

        if self.config.algorithm not in self._VALID_ALGO:
            raise ValueError(f"algorithm must be one of {sorted(self._VALID_ALGO)}")
        if self.config.distance not in self._VALID_DISTANCE:
            raise ValueError(f"distance must be one of {sorted(self._VALID_DISTANCE)}")
        if self.config.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self._is_fitted: bool = False
        self._feature_dim: Optional[int] = None
        self._classes: Optional[np.ndarray] = None

        # Stored parameters for nearest centroid.
        self._centroids: Optional[np.ndarray] = None

        # Stored training memory for kNN.
        self._X_train_mem: Optional[np.ndarray] = None
        self._y_train_mem: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "HSIModelTrainer":
        """Fit classifier from training features and labels."""
        X = _validate_2d_array(X, name="X")
        y = _validate_1d_labels(y, expected_len=X.shape[0], name="y")

        self._feature_dim = X.shape[1]
        classes = np.unique(y)
        if classes.shape[0] < 2:
            raise ValueError("Training labels must contain at least 2 classes")
        self._classes = classes

        if self.config.algorithm == "nearest_centroid":
            self._fit_nearest_centroid(X, y, classes)
            self._X_train_mem = None
            self._y_train_mem = None
        elif self.config.algorithm == "knn":
            self._centroids = None
            self._X_train_mem = X.astype(np.float32, copy=True)
            self._y_train_mem = y.copy()
        else:
            raise RuntimeError("Unsupported algorithm configuration")

        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels for input features."""
        self._check_fitted()
        X = _validate_2d_array(X, name="X")
        self._check_feature_dim(X)

        if self.config.algorithm == "nearest_centroid":
            return self._predict_nearest_centroid(X)
        if self.config.algorithm == "knn":
            return self._predict_knn(X)

        raise RuntimeError("Unsupported algorithm configuration")

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return class probabilities estimated from distances.

        Distances are converted to similarity scores via negative distance softmax.
        """
        self._check_fitted()
        X = _validate_2d_array(X, name="X")
        self._check_feature_dim(X)

        if self._classes is None:
            raise RuntimeError("Internal state error: missing class labels")

        if self.config.algorithm == "nearest_centroid":
            distances = _pairwise_distance(
                X,
                self._centroids_required(),
                metric=self.config.distance,
                eps=self.config.eps,
            )
            return _distance_to_probability(distances, eps=self.config.eps)

        # kNN probability by vote distribution of top-k neighbors.
        probs = np.zeros((X.shape[0], self._classes.shape[0]), dtype=np.float32)
        y_pred = self._predict_knn(X, return_neighbor_indices=True)
        neighbor_indices = y_pred[1]
        train_labels = self._y_train_required()

        for i in range(X.shape[0]):
            nn_labels = train_labels[neighbor_indices[i]]
            for cls_idx, cls in enumerate(self._classes):
                probs[i, cls_idx] = np.mean(nn_labels == cls)

        return probs

    def evaluate(self, X: ArrayLike, y_true: ArrayLike) -> Dict[str, object]:
        """Evaluate model and return metrics dictionary."""
        y_true = _validate_1d_labels(y_true, expected_len=None, name="y_true")
        y_pred = self.predict(X)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Prediction length mismatch: pred={y_pred.shape[0]}, true={y_true.shape[0]}"
            )

        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        prf = precision_recall_f1(y_true, y_pred, labels=labels, eps=self.config.eps)

        metrics: Dict[str, object] = {
            "accuracy": float(np.mean(y_true == y_pred)),
            "macro_precision": float(np.mean(prf["precision"])),
            "macro_recall": float(np.mean(prf["recall"])),
            "macro_f1": float(np.mean(prf["f1"])),
            "labels": labels,
            "confusion_matrix": cm,
            "per_class_precision": prf["precision"],
            "per_class_recall": prf["recall"],
            "per_class_f1": prf["f1"],
            "support": prf["support"],
        }
        return metrics

    @property
    def classes_(self) -> np.ndarray:
        """Sorted unique class labels seen during fit."""
        self._check_fitted()
        if self._classes is None:
            raise RuntimeError("Internal state error: missing class labels")
        return self._classes.copy()

    def _fit_nearest_centroid(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> None:
        centroids = []
        for cls in classes:
            cls_rows = X[y == cls]
            if cls_rows.shape[0] == 0:
                raise RuntimeError(f"No samples found for class {cls} during fit")
            centroids.append(cls_rows.mean(axis=0))
        self._centroids = np.stack(centroids).astype(np.float32, copy=False)

    def _predict_nearest_centroid(self, X: np.ndarray) -> np.ndarray:
        classes = self._classes_required()
        centroids = self._centroids_required()
        distances = _pairwise_distance(X, centroids, metric=self.config.distance, eps=self.config.eps)
        nearest = np.argmin(distances, axis=1)
        return classes[nearest]

    def _predict_knn(
        self,
        X: np.ndarray,
        return_neighbor_indices: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        classes = self._classes_required()
        X_train = self._X_train_required()
        y_train = self._y_train_required()

        k = min(self.config.k_neighbors, X_train.shape[0])
        pred = np.empty(X.shape[0], dtype=y_train.dtype)
        neighbor_idx_all = np.empty((X.shape[0], k), dtype=np.int64)

        for start in range(0, X.shape[0], self.config.batch_size):
            end = min(start + self.config.batch_size, X.shape[0])
            distances = _pairwise_distance(
                X[start:end],
                X_train,
                metric=self.config.distance,
                eps=self.config.eps,
            )
            nn_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            neighbor_idx_all[start:end] = nn_idx

            for i, row_idx in enumerate(range(start, end)):
                labels_k = y_train[nn_idx[i]]
                pred[row_idx] = _majority_vote(labels_k, classes)

        if return_neighbor_indices:
            return pred, neighbor_idx_all
        return pred

    def _check_feature_dim(self, X: np.ndarray) -> None:
        if self._feature_dim is None:
            raise RuntimeError("Internal state error: missing feature dimension")
        if X.shape[1] != self._feature_dim:
            raise ValueError(
                "Input feature dimension does not match fitted model: "
                f"got {X.shape[1]}, expected {self._feature_dim}"
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("HSIModelTrainer is not fitted. Call fit() first.")

    def _classes_required(self) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("Internal state error: missing class labels")
        return self._classes

    def _centroids_required(self) -> np.ndarray:
        if self._centroids is None:
            raise RuntimeError("Nearest centroid model is not fitted")
        return self._centroids

    def _X_train_required(self) -> np.ndarray:
        if self._X_train_mem is None:
            raise RuntimeError("kNN model is not fitted")
        return self._X_train_mem

    def _y_train_required(self) -> np.ndarray:
        if self._y_train_mem is None:
            raise RuntimeError("kNN model is not fitted")
        return self._y_train_mem


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute confusion matrix with rows=true and cols=pred."""
    y_true = _validate_1d_labels(y_true, expected_len=None, name="y_true")
    y_pred = _validate_1d_labels(y_pred, expected_len=y_true.shape[0], name="y_pred")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((labels.shape[0], labels.shape[0]), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


def precision_recall_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Per-class precision/recall/F1 and support."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm).astype(np.float32)
    support = cm.sum(axis=1).astype(np.int64)
    pred_count = cm.sum(axis=0).astype(np.float32)

    precision = tp / np.maximum(pred_count, eps)
    recall = tp / np.maximum(support.astype(np.float32), eps)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def _pairwise_distance(Xa: np.ndarray, Xb: np.ndarray, metric: str, eps: float) -> np.ndarray:
    if metric == "euclidean":
        a2 = np.sum(Xa * Xa, axis=1, keepdims=True)
        b2 = np.sum(Xb * Xb, axis=1, keepdims=True).T
        ab = Xa @ Xb.T
        d2 = np.maximum(a2 + b2 - 2.0 * ab, 0.0)
        return np.sqrt(d2, dtype=np.float32)

    if metric == "cosine":
        Xa_norm = np.linalg.norm(Xa, axis=1, keepdims=True)
        Xb_norm = np.linalg.norm(Xb, axis=1, keepdims=True)
        Xa_safe = Xa / np.maximum(Xa_norm, eps)
        Xb_safe = Xb / np.maximum(Xb_norm, eps)
        sim = Xa_safe @ Xb_safe.T
        return 1.0 - np.clip(sim, -1.0, 1.0)

    raise ValueError(f"Unsupported distance metric: {metric}")


def _distance_to_probability(distances: np.ndarray, eps: float) -> np.ndarray:
    # Convert distance to similarity logits, then softmax.
    logits = -distances
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.maximum(exp.sum(axis=1, keepdims=True), eps)
    return (exp / denom).astype(np.float32, copy=False)


def _majority_vote(labels_k: np.ndarray, all_classes: np.ndarray) -> np.generic:
    counts = np.array([(labels_k == c).sum() for c in all_classes], dtype=np.int64)
    return all_classes[int(np.argmax(counts))]


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
    "ModelConfig",
    "HSIModelTrainer",
    "confusion_matrix",
    "precision_recall_f1",
]
