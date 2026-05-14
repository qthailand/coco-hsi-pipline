from __future__ import annotations

import inspect
import json
import os
import glob
import re
import warnings
import traceback
from datetime import datetime
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle
from io import BytesIO

import cv2
import numpy as np
import spectral as spy
from matplotlib import colormaps
from matplotlib import cm
from matplotlib import colors as mcolors
import matplotlib.path as mpath
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D)

from PyQt5.QtCore import QObject, QPoint, Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
)
try:
    from sklearn.base import ClassifierMixin
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.inspection import permutation_importance
    from sklearn.utils import all_estimators
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install with: pip install scikit-learn") from exc

from feature_extraction import FeatureExtractionConfig, HSIFeatureExtractor
from hsi_loader import HSI2D_loader, _resolve_hsi_header_path, _load_coco_json_masks
from model_training import confusion_matrix, precision_recall_f1
from preprocessing import (
    HSIPreprocessor,
    PreprocessConfig,
    stratified_train_val_split,
    encode_str_labels,
    samplewise_dev_test_split,
)

# Module logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class _GuiLogEmitter(QObject):
    new_log = pyqtSignal(str, str)


class _QtLogHandler(logging.Handler):
    def __init__(self, emitter: _GuiLogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname
            # Emit via Qt signal so slot runs in main thread
            self.emitter.new_log.emit(level, msg)
        except Exception:
            self.handleError(record)


# Create a global emitter and attach handler to module logger
_log_emitter = _GuiLogEmitter()
_qt_handler = _QtLogHandler(_log_emitter)
_qt_handler.setLevel(logging.DEBUG)
_qt_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(_qt_handler)
# Also attach handler to root logger so other modules' INFO logs appear in the GUI
root_logger = logging.getLogger()
root_logger.addHandler(_qt_handler)
root_logger.setLevel(logging.INFO)


RGB_TARGET_WAVELENGTHS = (645.0, 555.0, 465.0)


@dataclass
class PipelineInput:
    dataset_path: str
    max_spectra: int
    val_ratio: float
    test_ratio: float
    spectral_norm: str
    global_scale: str
    feature_mode: str
    pca_components: Optional[float | int]
    selected_models: list[str]
    k_neighbors: int
    preview_sample_name: Optional[str]


class PipelineWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg: PipelineInput) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            result = self._run_pipeline(self.cfg)
            self.finished.emit(result)
        except Exception as exc:
            err = f"{exc}\n\n{traceback.format_exc()}"
            self.failed.emit(err)

    def _run_pipeline(self, cfg: PipelineInput) -> dict[str, object]:
        self.progress.emit("Loading HSI data...")
        start_time = time.time()
        model_timings: dict[str, float] = {}
        spectra, raw_labels, source_files, wavelengths = HSI2D_loader(
            dataset_path=cfg.dataset_path,
            max_spectra=cfg.max_spectra,
        )
        labels, label_names, label_order = encode_str_labels(raw_labels)

        self.progress.emit("Splitting data into train/val/test...")
        dev_idx, test_idx = samplewise_dev_test_split(
            labels=labels,
            source_files=source_files,
            test_ratio=cfg.test_ratio,
            random_state=42,
        )
        dev_labels = labels[dev_idx]
        train_local_idx, val_local_idx = stratified_train_val_split(
            dev_labels,
            val_ratio=cfg.val_ratio,
            random_state=42,
        )
        train_idx = dev_idx[train_local_idx]
        val_idx = dev_idx[val_local_idx]

        X_train_raw, y_train = spectra[train_idx], labels[train_idx]
        X_val_raw, y_val = spectra[val_idx], labels[val_idx]
        X_test_raw, y_test = spectra[test_idx], labels[test_idx]

        self.progress.emit("Preprocessing spectra...")
        pre_cfg = PreprocessConfig(
            spectral_normalization=cfg.spectral_norm,
            global_scaling=cfg.global_scale,
            remove_wavelength_ranges=[],
            clip_percentile=None,
        )
        pre = HSIPreprocessor(pre_cfg)
        X_train_pre = pre.fit_transform(X_train_raw, wavelengths=wavelengths)
        X_val_pre = pre.transform(X_val_raw)
        X_test_pre = pre.transform(X_test_raw)

        self.progress.emit("Extracting features...")
        feat_cfg = FeatureExtractionConfig(
            mode=cfg.feature_mode,
            pca_components=cfg.pca_components,
            center=True,
            scale=False,
        )
        feat = HSIFeatureExtractor(feat_cfg)
        X_train_feat = feat.fit_transform(X_train_pre)
        X_val_feat = feat.transform(X_val_pre)
        X_test_feat = feat.transform(X_test_pre)

        self.progress.emit("Preparing preview sample...")
        sample_name, sample_rgb, gt_map, sample_feat = _prepare_sample_preview_data(
            dataset_path=cfg.dataset_path,
            preprocessor=pre,
            feature_extractor=feat,
            sample_name=cfg.preview_sample_name,
        )

        # remap gt_map: raw COCO cat_id → encoded int (ผ่านชื่อ class)
        gt_json_path = os.path.join(cfg.dataset_path, "GT", f"{sample_name}.json")
        preview_cat_names = _load_preview_category_names(gt_json_path)
        name_to_encoded = {name: enc for enc, name in label_names.items()}
        gt_map_encoded = np.zeros_like(gt_map, dtype=np.int32)
        for raw_id, class_name in preview_cat_names.items():
            enc_id = name_to_encoded.get(class_name)
            if enc_id is not None:
                gt_map_encoded[gt_map == raw_id] = enc_id

        model_specs = _build_sklearn_model_specs(
            selected_model_names=cfg.selected_models,
            k_neighbors=cfg.k_neighbors,
        )
        if not model_specs:
            raise RuntimeError(
                "No usable selected scikit-learn classifiers found")

        metric_labels = _build_metric_label_order(label_order, labels)

        results: list[dict[str, object]] = []
        failures: list[str] = []
        best_score = -1.0
        best_name = ""
        best_estimator: Optional[ClassifierMixin] = None
        best_pred_map = np.zeros_like(gt_map, dtype=np.int32)

        total = len(model_specs)
        for idx, (name, estimator) in enumerate(model_specs, start=1):
            self.progress.emit(f"Training [{idx}/{total}] {name}...")
            model_start = time.time()
            try:
                logger.info("Starting training: %s (%d/%d)", name, idx, total)
                logger.debug("X_train_feat.shape=%s, y_train.shape=%s", getattr(
                    X_train_feat, 'shape', None), getattr(y_train, 'shape', None))
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", category=ConvergenceWarning)
                    estimator.fit(X_train_feat, y_train)
                logger.info("Finished fit for %s", name)
                y_val_pred = estimator.predict(X_val_feat)
                val_metrics = _compute_metrics(
                    y_val, y_val_pred, metric_labels)
                y_test_pred = estimator.predict(X_test_feat)
                test_metrics = _compute_metrics(
                    y_test, y_test_pred, metric_labels)

                # Capture feature importances (tree-based) or linear coefficients
                fi: Optional[np.ndarray] = None
                if hasattr(estimator, "feature_importances_"):
                    fi = np.asarray(
                        estimator.feature_importances_, dtype=np.float32)
                elif hasattr(estimator, "coef_"):
                    coef = np.asarray(estimator.coef_, dtype=np.float32)
                    fi = np.abs(coef).mean(
                        axis=0) if coef.ndim > 1 else np.abs(coef)
                else:
                    # Fallback: model-agnostic importance so all models can be visualized
                    try:
                        logger.info(
                            "Computing permutation importance for %s", name)
                        perm = permutation_importance(
                            estimator,
                            X_val_feat,
                            y_val,
                            scoring="f1_macro",
                            n_repeats=4,
                            random_state=42,
                            n_jobs=1,
                        )
                        fi = np.asarray(perm.importances_mean,
                                        dtype=np.float32)
                        logger.info(
                            "Permutation importance computed for %s", name)
                    except Exception:
                        logger.exception(
                            "Permutation importance failed for %s", name)
                        fi = None

                # Force an array for every model so FI page is always available
                if fi is None or fi.size == 0:
                    fi = np.zeros(X_train_feat.shape[1], dtype=np.float32)

                elapsed = time.time() - model_start
                results.append(
                    {
                        "name": name,
                        "metrics": test_metrics,
                        "val_metrics": val_metrics,
                        "feature_importances": fi,
                        "time_sec": float(elapsed),
                    }
                )
                model_timings[str(name)] = float(elapsed)

                score = float(val_metrics["macro_f1"])
                if score > best_score:
                    pred_pixels = estimator.predict(sample_feat)
                    pred_map = pred_pixels.reshape(
                        gt_map.shape).astype(np.int32, copy=False)
                    # Mask out background/unannotated pixels (gt == 0)
                    pred_map[gt_map == 0] = 0
                    best_pred_map = pred_map
                    best_score = score
                    best_name = name
                    best_estimator = estimator
            except Exception as exc:
                logger.exception("Training failed for %s", name)
                failures.append(f"{name}: {exc}")

        if not results:
            detail = "\n".join(
                failures[:8]) if failures else "No model-specific error captured."
            raise RuntimeError(
                "All selected models failed. Try different models or simpler features.\n\n"
                f"Details:\n{detail}"
            )

        results.sort(key=lambda x: float(
            x.get("val_metrics", x["metrics"])["macro_f1"]), reverse=True)
        if not best_name:
            best_name = str(results[0]["name"])
        if best_estimator is None:
            raise RuntimeError(
                "Internal error: best estimator was not captured")

        # Evaluate only on annotated pixels (gt != 0)
        _ann_flat = gt_map.reshape(-1) != 0
        if _ann_flat.any():
            preview_metrics = _compute_metrics(
                gt_map_encoded.reshape(-1)[_ann_flat],
                best_pred_map.reshape(-1)[_ann_flat],
                metric_labels)
        else:
            preview_metrics = _compute_metrics(
                gt_map_encoded.reshape(-1), best_pred_map.reshape(-1), metric_labels)
        split_reports = _build_split_reports(
            dataset_path=cfg.dataset_path,
            preprocessor=pre,
            feature_extractor=feat,
            source_files=source_files,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            X_train_feat=X_train_feat,
            X_val_feat=X_val_feat,
            X_test_feat=X_test_feat,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            estimator=best_estimator,
            metric_labels=metric_labels,
            best_model_name=best_name,
            label_names=label_names,
        )

        total_time = time.time() - start_time
        feat_names = _build_feature_names(
            cfg.feature_mode, int(X_train_feat.shape[1]), wavelengths)
        return {
            "n_train": int(X_train_feat.shape[0]),
            "n_val": int(X_val_feat.shape[0]),
            "n_test": int(X_test_feat.shape[0]),
            "feature_dim": int(X_train_feat.shape[1]),
            "label_names": label_names,
            "feature_names": feat_names,
            "split_stats": _compute_split_stats(y_train, y_test),
            "source_split_rows": _compute_source_split_rows(source_files, train_idx, val_idx, test_idx),
            "split_preview": _compute_split_preview_payload(X_train_feat, y_train, X_test_feat, y_test),
            "raw_spectrum_stats": _compute_raw_spectrum_stats(X_train_raw, y_train, X_test_raw, y_test, wavelengths),
            "model_results": results,
            "model_timings": model_timings,
            "total_time": float(total_time),
            "failed_models": failures,
            "sample_name": sample_name,
            "sample_rgb": sample_rgb,
            "gt_map": gt_map,
            "best_model_name": best_name,
            "best_pred_map": best_pred_map,
            "preview_metrics": preview_metrics,
            "split_reports": split_reports,
        }


class MacTitleBar(QWidget):
    def __init__(self, parent_window: "HSIPipelineWindow") -> None:
        super().__init__(parent_window)
        self._window = parent_window
        self._drag_pos: Optional[QPoint] = None

        self.setObjectName("titleBar")
        self.setFixedHeight(38)
        self.setMouseTracking(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        self.close_btn = QPushButton("")
        self.close_btn.setObjectName("macClose")
        self.close_btn.setFixedSize(12, 12)
        self.close_btn.clicked.connect(self._window.close)
        self.close_btn.setCursor(Qt.ArrowCursor)
        self.close_btn.setToolTip("Close")

        self.min_btn = QPushButton("")
        self.min_btn.setObjectName("macMin")
        self.min_btn.setFixedSize(12, 12)
        self.min_btn.clicked.connect(self._window.showMinimized)
        self.min_btn.setCursor(Qt.ArrowCursor)
        self.min_btn.setToolTip("Minimize")

        self.max_btn = QPushButton("")
        self.max_btn.setObjectName("macMax")
        self.max_btn.setFixedSize(12, 12)
        self.max_btn.clicked.connect(self._toggle_max_restore)
        self.max_btn.setCursor(Qt.ArrowCursor)
        self.max_btn.setToolTip("Zoom")

        self.title_label = QLabel("HSI Classification - Model Runner")
        self.title_label.setObjectName("windowTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.control_cluster = QWidget()
        self.control_cluster.setObjectName("macControlCluster")
        self.control_cluster.setMouseTracking(True)

        left_layout = QHBoxLayout(self.control_cluster)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(7)
        left_layout.addWidget(self.close_btn)
        left_layout.addWidget(self.min_btn)
        left_layout.addWidget(self.max_btn)

        layout.addWidget(self.control_cluster, 0, Qt.AlignLeft)
        layout.addStretch(1)
        layout.addWidget(self.title_label, 0, Qt.AlignCenter)
        layout.addStretch(1)
        spacer = QWidget()
        spacer.setFixedWidth(46)
        layout.addWidget(spacer)

    def _toggle_max_restore(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self._window.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton and self._drag_pos is not None and not self._window.isMaximized():
            self._window.move(event.globalPos() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._toggle_max_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class HSIPipelineWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HSI Classification - Model Runner")
        self.resize(1320, 860)
        self.setMinimumSize(1020, 680)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._available_model_entries = _list_available_model_entries(
            k_neighbors=5)
        self._available_model_names = [
            name for _, name in self._available_model_entries]
        self._checked_model_names: set[str] = set()
        self._thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None
        self._canvases: list[FigureCanvasQTAgg] = []

        # preview hover and annotation state
        self._preview_ann_info: list[dict] = []
        self._preview_hover_patch = None
        self._preview_hover_im = None
        self._preview_hover_text = None
        self._preview_current_ann_id = None
        self._preview_hsi_cube = None
        self._figures: list[Figure] = []
        self._last_result: Optional[dict[str, object]] = None
        self._last_run_config: Optional[PipelineInput] = None

        self._dark_mode = False
        self._build_ui()
        self._apply_style()
        self._set_theme(is_dark_mode())
        # Initialize log widget reference
        self._info_log_widget: Optional[QTextEdit] = None
        # Connect global logger emitter to window append method
        _log_emitter.new_log.connect(self._append_log)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("rootWrap")
        self.setCentralWidget(root)
        main_layout = QGridLayout(root)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setHorizontalSpacing(0)

        shell = QFrame()
        shell.setObjectName("windowShell")
        shell.setGraphicsEffect(self._build_shadow_effect())
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self.title_bar = MacTitleBar(self)
        shell_layout.addWidget(self.title_bar)

        content = QWidget()
        content_layout = QGridLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setHorizontalSpacing(12)

        left_card = QFrame()
        left_card.setObjectName("card")
        left_card.setMinimumWidth(430)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        title = QLabel("Pipeline Settings")
        title.setObjectName("cardTitle")
        left_layout.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout(dataset_row)
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        dataset_row_layout.setSpacing(6)
        self.dataset_edit = QLineEdit(os.path.join(os.getcwd(), "Dataset"))
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dataset)
        dataset_row_layout.addWidget(self.dataset_edit)
        dataset_row_layout.addWidget(browse_btn)
        dataset_row_layout.setStretch(0, 1)

        preview_row = QWidget()
        preview_row_layout = QHBoxLayout(preview_row)
        preview_row_layout.setContentsMargins(0, 0, 0, 0)
        preview_row_layout.setSpacing(6)
        self.preview_sample_combo = QComboBox()
        self.preview_sample_combo.setObjectName("previewSampleCombo")
        self.refresh_preview_btn = QPushButton("Refresh")
        self.refresh_preview_btn.clicked.connect(self._refresh_preview_samples)
        preview_row_layout.addWidget(self.preview_sample_combo)
        preview_row_layout.addWidget(self.refresh_preview_btn)
        preview_row_layout.setStretch(0, 1)

        self.max_spectra_edit = QLineEdit("1000")
        self.val_ratio_edit = QLineEdit("0.2")
        self.test_ratio_edit = QLineEdit("0.5")

        self.spectral_norm_combo = QComboBox()
        self.spectral_norm_combo.addItems(["none", "snv", "l2", "max", "area"])
        self.spectral_norm_combo.setCurrentText("snv")

        self.global_scale_combo = QComboBox()
        self.global_scale_combo.addItems(["none", "standard", "minmax"])
        self.global_scale_combo.setCurrentText("standard")

        self.feature_mode_combo = QComboBox()
        self.feature_mode_combo.addItems(["raw", "stat", "pca", "pca_stat"])
        self.feature_mode_combo.setCurrentText("stat")
        self.pca_comp_edit = QLineEdit("0.99")

        self.k_neighbors_edit = QLineEdit("5")

        form.addRow("Dataset Path", dataset_row)
        form.addRow("Preview Sample", preview_row)
        left_layout.addLayout(form)

        settings_grid = QGridLayout()
        settings_grid.setContentsMargins(0, 0, 0, 0)
        settings_grid.setHorizontalSpacing(12)
        settings_grid.setVerticalSpacing(8)

        settings_grid.addWidget(QLabel("Max Spectra / Sample"), 0, 0)
        settings_grid.addWidget(self.max_spectra_edit, 0, 1)

        settings_grid.addWidget(QLabel("Validation Ratio"), 1, 0)
        settings_grid.addWidget(self.val_ratio_edit, 1, 1)
        settings_grid.addWidget(QLabel("Train/Test Ratio"), 1, 2)
        settings_grid.addWidget(self.test_ratio_edit, 1, 3)

        settings_grid.addWidget(QLabel("Spectral Normalization"), 2, 0)
        settings_grid.addWidget(self.spectral_norm_combo, 2, 1)
        settings_grid.addWidget(QLabel("Global Scaling"), 2, 2)
        settings_grid.addWidget(self.global_scale_combo, 2, 3)

        settings_grid.addWidget(QLabel("Feature Mode"), 3, 0)
        settings_grid.addWidget(self.feature_mode_combo, 3, 1)
        settings_grid.addWidget(QLabel("PCA Components"), 3, 2)
        settings_grid.addWidget(self.pca_comp_edit, 3, 3)

        settings_grid.addWidget(QLabel("k Neighbors (KNN)"), 4, 0)
        settings_grid.addWidget(self.k_neighbors_edit, 4, 1)

        left_layout.addLayout(settings_grid)

        model_group = QGroupBox("Models")
        model_group_layout = QVBoxLayout(model_group)
        model_group_layout.setContentsMargins(10, 10, 10, 10)
        model_group_layout.setSpacing(8)

        model_actions = QWidget()
        model_actions_layout = QHBoxLayout(model_actions)
        model_actions_layout.setContentsMargins(0, 0, 0, 0)
        model_actions_layout.setSpacing(6)
        self.select_all_models_btn = QPushButton("Select All")
        self.clear_models_btn = QPushButton("Clear")
        self.select_all_models_btn.clicked.connect(self._select_all_models)
        self.clear_models_btn.clicked.connect(self._clear_model_selection)
        model_actions_layout.addWidget(self.select_all_models_btn)
        model_actions_layout.addWidget(self.clear_models_btn)
        model_actions_layout.addStretch(1)

        self.model_category_combo = QComboBox()
        self.model_category_combo.addItem("All")
        for category in sorted({category for category, _ in self._available_model_entries}):
            self.model_category_combo.addItem(category)
        self.model_category_combo.currentTextChanged.connect(
            self._populate_model_list)

        self.model_list = QListWidget()
        self.model_list.setObjectName("modelList")
        self.model_list.setMinimumHeight(280)
        self.model_list.itemChanged.connect(self._on_model_item_changed)

        self._select_default_models()
        self._populate_model_list("All")

        model_group_layout.addWidget(model_actions)
        model_group_layout.addWidget(self.model_category_combo)
        model_group_layout.addWidget(self.model_list)
        left_layout.addWidget(model_group)

        self.theme_toggle_btn = QPushButton("Toggle Dark/Light Mode")
        self.theme_toggle_btn.setObjectName("themeToggle")
        self.theme_toggle_btn.clicked.connect(self._on_toggle_theme)
        left_layout.addWidget(self.theme_toggle_btn)

        self.run_btn = QPushButton("Run Selected Models")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run_pipeline_async)
        left_layout.addWidget(self.run_btn)

        self.export_pdf_btn = QPushButton("Export Report PDF")
        self.export_pdf_btn.clicked.connect(self._export_report_pdf)
        self.export_pdf_btn.setEnabled(False)
        left_layout.addWidget(self.export_pdf_btn)

        hint = QLabel(
            "Tip: เลือกเฉพาะโมเดลที่ต้องการรันได้เลย หรือกด Select All")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        left_layout.addWidget(hint)
        left_layout.addStretch(1)

        # Data processing visualization buttons (2x3 grid)
        proc_group = QGroupBox("Data Processing")
        proc_layout = QGridLayout(proc_group)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setHorizontalSpacing(8)
        proc_layout.setVerticalSpacing(6)

        self.btn_raw_spectra = QPushButton("Raw Spectra")
        self.btn_raw_spectra.clicked.connect(self._on_show_raw_spectra)
        self.btn_mean_std = QPushButton("Mean ± Std")
        self.btn_mean_std.clicked.connect(self._on_show_mean_std)
        self.btn_2d_feat = QPushButton("2D Feature Viz")
        self.btn_2d_feat.clicked.connect(self._on_show_2d_feature_viz)
        self.btn_3d_feat = QPushButton("3D Feature Viz")
        self.btn_3d_feat.clicked.connect(self._on_show_3d_feature_viz)
        self.btn_dataset_preview = QPushButton("Dataset Preview")
        self.btn_dataset_preview.clicked.connect(self._on_show_dataset_preview)
        self.btn_norm_plot = QPushButton("Normalize Plot")
        self.btn_norm_plot.clicked.connect(self._on_show_normalize_plot)
        self.btn_global_scale = QPushButton("Global Scale Plot")
        self.btn_global_scale.clicked.connect(self._on_show_global_scale_plot)

        # Optional: make buttons visually consistent size
        for btn in (self.btn_raw_spectra, self.btn_mean_std, self.btn_2d_feat, self.btn_3d_feat, self.btn_norm_plot, self.btn_global_scale):
            btn.setMinimumHeight(30)

        # Place buttons in 2 rows x 3 columns
        proc_layout.addWidget(self.btn_raw_spectra, 0, 0)
        proc_layout.addWidget(self.btn_mean_std, 0, 1)
        proc_layout.addWidget(self.btn_2d_feat, 0, 2)
        proc_layout.addWidget(self.btn_3d_feat, 1, 0)
        proc_layout.addWidget(self.btn_dataset_preview, 1, 1)
        proc_layout.addWidget(self.btn_norm_plot, 1, 2)
        proc_layout.addWidget(self.btn_global_scale, 2, 0, 1, 3)

        left_layout.addWidget(proc_group)

        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(8)

        result_title = QLabel("Results")
        result_title.setObjectName("cardTitle")
        right_layout.addWidget(result_title)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        right_layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("resultTabs")
        right_layout.addWidget(self.tabs, 1)
        self._add_placeholder_tab()
        self._refresh_preview_samples()

        content_layout.addWidget(left_card, 0, 0)
        content_layout.addWidget(right_card, 0, 1)
        content_layout.setColumnStretch(0, 5)
        content_layout.setColumnStretch(1, 7)

        shell_layout.addWidget(content, 1)
        main_layout.addWidget(shell, 0, 0)

    def _build_shadow_effect(self) -> QGraphicsDropShadowEffect:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 55))
        return shadow

    def _apply_style(self) -> None:
        # ค้นหา gui_macos.qss ใน pipeline/ ก่อน ถ้าไม่มีให้ขึ้นไป project root
        script_dir = Path(__file__).parent
        qss_path = script_dir / "gui_macos.qss"
        if not qss_path.exists():
            qss_path = script_dir.parent / "gui_macos.qss"
        app = QApplication.instance()
        if qss_path.exists():
            app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
            return

        # Fallback style in case qss file is missing.
        app.setStyleSheet(
            "QWidget { background: #f5f5f7; color: #1d1d1f; }"
            "QFrame#card { background: #ffffff; border: 1px solid #e6e6eb; border-radius: 0px; }"
        )

    def _set_theme(self, dark: bool) -> None:
        self._dark_mode = dark
        apply_theme(QApplication.instance(), dark)
        if hasattr(self, "theme_toggle_btn"):
            self.theme_toggle_btn.setText(
                "Switch to Light Mode" if dark else "Switch to Dark Mode"
            )

    def _on_toggle_theme(self) -> None:
        self._set_theme(not self._dark_mode)

    def _browse_dataset(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", self.dataset_edit.text().strip())
        if selected:
            self.dataset_edit.setText(selected)
            self._refresh_preview_samples()

    def _refresh_preview_samples(self) -> None:
        dataset = self.dataset_edit.text().strip()
        samples = _list_dataset_pair_names(dataset)

        self.preview_sample_combo.blockSignals(True)
        self.preview_sample_combo.clear()
        self.preview_sample_combo.addItem("Auto (first)", None)
        for sample in samples:
            self.preview_sample_combo.addItem(sample, sample)
        self.preview_sample_combo.setCurrentIndex(0)
        self.preview_sample_combo.blockSignals(False)

    def _parse_pca_components(self, raw: str) -> Optional[int | float]:
        text = raw.strip().lower()
        if text in {"", "none", "auto"}:
            return None
        if "." in text:
            return float(text)
        return int(text)

    def _selected_model_names(self) -> list[str]:
        return [name for name in self._available_model_names if name in self._checked_model_names]

    def _select_default_models(self) -> None:
        self._checked_model_names = {
            "LogisticRegression",
            "RandomForestClassifier",
            "SVC",
            "KNeighborsClassifier",
            "GaussianNB",
        }

    def _select_all_models(self) -> None:
        current_category = self.model_category_combo.currentText()
        for category, model_name in self._available_model_entries:
            if current_category == "All" or category == current_category:
                self._checked_model_names.add(model_name)
        self._populate_model_list(current_category)

    def _clear_model_selection(self) -> None:
        current_category = self.model_category_combo.currentText()
        if current_category == "All":
            self._checked_model_names.clear()
        else:
            for category, model_name in self._available_model_entries:
                if category == current_category:
                    self._checked_model_names.discard(model_name)
        self._populate_model_list(current_category)

    def _populate_model_list(self, category: str) -> None:
        self.model_list.blockSignals(True)
        self.model_list.clear()
        for model_category, model_name in self._available_model_entries:
            if category != "All" and model_category != category:
                continue
            item = QListWidgetItem(model_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable |
                          Qt.ItemIsEnabled)
            item.setCheckState(
                Qt.Checked if model_name in self._checked_model_names else Qt.Unchecked)
            item.setData(Qt.UserRole, model_name)
            item.setToolTip(model_category)
            self.model_list.addItem(item)
        self.model_list.blockSignals(False)

    def _on_model_item_changed(self, item: QListWidgetItem) -> None:
        model_name = item.data(Qt.UserRole)
        if not model_name:
            return
        if item.checkState() == Qt.Checked:
            self._checked_model_names.add(str(model_name))
        else:
            self._checked_model_names.discard(str(model_name))

    def _build_config(self) -> PipelineInput:
        dataset = self.dataset_edit.text().strip()
        if not dataset:
            raise ValueError("Please set dataset path")
        selected_models = self._selected_model_names()
        if not selected_models:
            raise ValueError("Please select at least one model")

        return PipelineInput(
            dataset_path=dataset,
            max_spectra=int(self.max_spectra_edit.text().strip()),
            val_ratio=float(self.val_ratio_edit.text().strip()),
            test_ratio=float(self.test_ratio_edit.text().strip()),
            spectral_norm=self.spectral_norm_combo.currentText().strip(),
            global_scale=self.global_scale_combo.currentText().strip(),
            feature_mode=self.feature_mode_combo.currentText().strip(),
            pca_components=self._parse_pca_components(
                self.pca_comp_edit.text()),
            selected_models=selected_models,
            k_neighbors=int(self.k_neighbors_edit.text().strip()),
            preview_sample_name=self.preview_sample_combo.currentData(),
        )

    def _run_pipeline_async(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return

        try:
            cfg = self._build_config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.export_pdf_btn.setEnabled(False)
        self.status_label.setText("Running selected models...")
        self._clear_tabs()
        # Show live log in Info tab while running
        log_tab = self._create_log_tab()
        self.tabs.addTab(log_tab, "Info")
        self.tabs.setCurrentWidget(log_tab)
        self._last_result = None
        self._last_run_config = cfg

        self._thread = QThread(self)
        self._worker = PipelineWorker(cfg)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_pipeline_success)
        self._worker.failed.connect(self._on_pipeline_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _on_pipeline_success(self, result: dict[str, object]) -> None:
        self._clear_tabs()

        label_names = result.get("label_names") or {}

        summary = self._create_summary_tab(result, label_names)
        self.tabs.addTab(summary, "Summary")

        model_results = result["model_results"]
        for model_result in model_results:
            name = str(model_result["name"])
            metrics = model_result["metrics"]
            tab = self._create_model_tab(
                name, metrics, label_names)
            self.tabs.addTab(tab, name)

        failed_models = result.get("failed_models", [])
        status = (
            f"Done: {len(model_results)} success"
            + (f", {len(failed_models)} failed" if failed_models else "")
        )
        self.status_label.setText(status)
        self.run_btn.setEnabled(True)
        self.export_pdf_btn.setEnabled(True)
        self._last_result = result

    def _on_pipeline_error(self, err_text: str) -> None:
        self._clear_tabs()
        self._add_placeholder_tab("Pipeline failed. ดูรายละเอียดด้านล่าง")
        self.status_label.setText("Failed")
        self.run_btn.setEnabled(True)
        self.export_pdf_btn.setEnabled(False)
        QMessageBox.critical(self, "Pipeline Error", err_text.splitlines()[0])

    def _clear_tabs(self) -> None:
        self.tabs.clear()
        for canvas in self._canvases:
            canvas.setParent(None)
            canvas.deleteLater()
        self._canvases.clear()
        self._figures.clear()

    def _add_placeholder_tab(self, message: str = "Select models and run to see results.") -> None:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
        self.tabs.addTab(holder, "Info")

    def _append_log(self, level: str, msg: str) -> None:
        # Append a log message to the Info log widget if present
        try:
            text = getattr(self, "_info_log_widget", None)
            if text is None:
                return
            color = "#000000"
            lvl = (level or "").upper()
            if lvl == "WARNING" or lvl == "WARN":
                color = "#ff8800"
            elif lvl == "ERROR" or lvl == "CRITICAL":
                color = "#cc0000"
            safe = msg.replace("&", "&amp;").replace(
                "<", "&lt;").replace(">", "&gt;")
            html = f'<div><span style="color:{color}">{safe}</span></div>'
            text.moveCursor(text.textCursor().End)
            text.insertHtml(html)
            text.insertPlainText("\n")
            text.ensureCursorVisible()
        except Exception:
            # never raise from logging appender
            pass

    def _create_log_tab(self) -> QWidget:
        # Build a QTextEdit that displays logs with simple color rules
        holder = QWidget()
        layout = QVBoxLayout(holder)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QTextEdit.NoWrap)
        text.setFontFamily("Consolas")
        layout.addWidget(text)
        layout.addStretch(1)
        # Store reference for appending
        self._info_log_widget = text

        # The actual connection to the emitter is handled by the window instance
        return holder

    def _create_summary_tab(self, result: dict[str, object], label_names: dict[int, str]) -> QWidget:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        text.setLineWrapMode(QTextEdit.WidgetWidth)

        model_results = result["model_results"]
        best_name = str(result["best_model_name"])
        best_labels = self._get_best_labels(result, best_name)
        lines = self._build_summary_lines(result, best_labels, label_names)

        text.setPlainText("\n".join(lines))

        fig = self._build_summary_figure(
            sample_name=str(result["sample_name"]),
            sample_rgb=np.asarray(result["sample_rgb"]),
            pred_map=np.asarray(result["best_pred_map"]),
            preview_metrics=dict(result["preview_metrics"]),
            best_model_name=best_name,
            class_labels=best_labels,
            label_names=label_names,
            annotation_mask=np.asarray(result.get("gt_map", np.zeros_like(
                result.get("best_pred_map", np.zeros((1, 1), dtype=np.int32))))) != 0,
        )
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        canvas.draw()

        self._canvases.append(canvas)
        self._figures.append(fig)

        return self._build_result_tab(text, canvas)

    def _get_best_labels(self, result: dict[str, object], best_name: str) -> np.ndarray:
        for item in result["model_results"]:
            if str(item["name"]) == best_name:
                return np.asarray(item["metrics"]["labels"])
        return np.array([], dtype=np.int32)

    def _build_summary_lines(
        self,
        result: dict[str, object],
        best_labels: np.ndarray,
        label_names: dict[int, str],
    ) -> list[str]:
        best_name = str(result["best_model_name"])
        lines = [
            "Selected Models Summary",
            "=" * 56,
        ]

        # Add concise model count and names for quick overview
        try:
            model_names = [str(item.get("name", ""))
                           for item in result.get("model_results", [])]
            if model_names:
                lines.append(
                    f"Models used ({len(model_names)}): {', '.join(model_names)}"
                )
        except Exception:
            pass

        # Add timing summary (total + per-model if available)
        try:
            total_time = float(result.get("total_time", 0.0))
            lines.append(f"Total runtime: {total_time:.2f} s")
            model_timings = result.get("model_timings", {}) or {}
            if model_timings:
                for nm in model_names:
                    t = model_timings.get(str(nm))
                    if t is not None:
                        lines.append(f"  {nm}: {t:.2f} s")
        except Exception:
            pass

        lines.extend([
            f"Train samples : {result['n_train']}",
            f"Val samples   : {result['n_val']}",
            f"Test samples  : {result.get('n_test', 0)}",
            f"Feature dim   : {result['feature_dim']}",
            f"Best model    : {best_name}",
            "Model select  : validation split",
            "Test metrics  : test split",
            "Preview map   : preview sample",
            "",
            "Results by Macro F1 (on Test set)",
            "Model | Accuracy | Macro F1",
            "-" * 56,
        ])
        for item in result["model_results"][:10]:
            m = item["metrics"]
            lines.append(
                f"{item['name']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f}"
            )

        failed = result.get("failed_models", [])
        if failed:
            lines.extend(["", f"Failed models: {len(failed)}"])
            lines.extend(failed[:20])

        if best_labels.size > 0:
            class_values = ", ".join(
                _format_class_label(int(v), label_names) for v in best_labels.astype(np.int32, copy=False)
            )
            if class_values:
                lines.extend(["", f"Class labels: {class_values}"])
        return lines

    def _create_model_tab(
        self,
        name: str,
        metrics: dict[str, object],
        label_names: dict[int, str],
    ) -> QWidget:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        text.setLineWrapMode(QTextEdit.WidgetWidth)
        text.setPlainText(self._format_model_metrics_text(
            name, metrics, label_names))

        fig = self._build_confusion_figure(
            name, metrics, label_names)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        canvas.draw()

        self._canvases.append(canvas)
        self._figures.append(fig)

        return self._build_result_tab(text, canvas)

    # UserWarning: This figure was using a layout engine that is incompatible with subplots_adjust and/or tight_layout; not calling subplots_adjust.
    # fig.subplots_adjust(left=0.24, right=0.92, bottom=0.28, top=0.90)
    def _show_best_model_popup(self) -> None:
        if self._last_result is None:
            QMessageBox.information(
                self, "No Results", "Please run models before showing the best model.")
            return

        result = self._last_result
        best_name = str(result.get("best_model_name", "Best"))
        best_labels = self._get_best_labels(result, best_name)
        label_names = result.get("label_names", {})

        sample_name = str(result.get("sample_name", "N/A"))
        sample_rgb = np.asarray(result.get(
            "sample_rgb", np.zeros((32, 32, 3), dtype=np.float32)))
        pred_map = np.asarray(result.get(
            "best_pred_map", np.zeros((32, 32), dtype=np.int32)))
        preview_metrics = dict(result.get("preview_metrics", {}))

        fig = self._build_summary_figure(
            sample_name=sample_name,
            sample_rgb=sample_rgb,
            pred_map=pred_map,
            preview_metrics=preview_metrics,
            best_model_name=best_name,
            class_labels=best_labels,
            label_names=label_names,
            page_label="Best Model",
        )

        # ✅ ปิด layout engine ทุกชนิดอย่างชัดเจน
        fig.set_layout_engine("none")

        try:
            fig.show()
        except Exception:
            try:
                plt.show(block=False)
            except Exception:
                # As a last resort, embed into a canvas and show blocking
                canvas = FigureCanvasQTAgg(fig)
                canvas.draw()
                fig.canvas = canvas
                plt.show()

    def _show_figure_popup(self, fig: Figure) -> None:
        # Try to avoid moving the original Figure's canvas by cloning the Figure
        try:
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        except Exception:
            NavigationToolbar2QT = None

        # First attempt: deep-copy the Figure via pickle (preserves artists)
        cloned_fig = None
        try:
            cloned_fig = pickle.loads(pickle.dumps(fig))
        except Exception:
            cloned_fig = None

        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("Figure")
            dlg.resize(900, 700)
            layout = QVBoxLayout(dlg)

            if cloned_fig is not None:
                popup_fig = cloned_fig
                popup_canvas = FigureCanvasQTAgg(popup_fig)
                if NavigationToolbar2QT is not None:
                    try:
                        toolbar = NavigationToolbar2QT(popup_canvas, dlg)
                        layout.addWidget(toolbar)
                    except Exception:
                        pass
                layout.addWidget(popup_canvas)
                popup_canvas.draw()
                dlg.setAttribute(Qt.WA_DeleteOnClose)
                dlg.show()
                return

            # Fallback: render original figure to PNG bytes and show as image (no interactivity)
            try:
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                pix = QPixmap()
                pix.loadFromData(buf.getvalue(), "PNG")

                label = QLabel()
                label.setPixmap(pix)
                label.setScaledContents(True)

                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(label)

                layout.addWidget(scroll)
                dlg.setAttribute(Qt.WA_DeleteOnClose)
                dlg.show()
                return
            except Exception:
                pass

            # Last resort: non-blocking plt.show
            try:
                plt.show(block=False)
                return
            except Exception:
                QMessageBox.information(
                    self, "Show Figure", "Unable to open figure window.")
        except Exception:
            QMessageBox.information(
                self, "Show Figure", "Unable to open figure window.")

    def _build_result_tab(self, text_widget: QTextEdit, canvas: FigureCanvasQTAgg) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        text_widget.setMinimumWidth(280)
        splitter.addWidget(text_widget)

        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        # Add a small toolbar with an "Open Figure" button for popup viewing
        toolbar = QWidget()
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(0, 0, 0, 0)
        tb_layout.setSpacing(6)
        open_btn = QPushButton("Open Figure")

        def _on_open():
            try:
                fig = canvas.figure
                self._show_figure_popup(fig)
            except Exception as exc:
                QMessageBox.information(
                    self, "Show Figure", f"Failed to open figure:\n{exc}")
        open_btn.clicked.connect(_on_open)
        tb_layout.addWidget(open_btn)
        tb_layout.addStretch(1)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)

        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_scroll.setWidget(plot_host)
        plot_scroll.setMinimumWidth(320)
        splitter.addWidget(plot_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([360, 640])

        layout.addWidget(splitter)
        return tab

    # ---------- Data processing visualization handlers ----------
    def _load_sample_spectra(self, max_spectra: int = 2000):
        dataset = self.dataset_edit.text().strip()
        if not dataset:
            raise ValueError("Please set dataset path")
        max_spectra = max(1, int(max_spectra))
        spectra, raw_labels, source_files, wavelengths = HSI2D_loader(
            dataset_path=dataset, max_spectra=max_spectra
        )
        labels, label_names, _ = encode_str_labels(raw_labels)
        return spectra, labels, source_files, wavelengths, label_names

    def _build_preprocessor_and_extractor(self):
        pre_cfg = PreprocessConfig(
            spectral_normalization=self.spectral_norm_combo.currentText().strip(),
            global_scaling=self.global_scale_combo.currentText().strip(),
            remove_wavelength_ranges=[],
            clip_percentile=None,
        )
        feat_cfg = FeatureExtractionConfig(
            mode=self.feature_mode_combo.currentText().strip(),
            pca_components=self._parse_pca_components(
                self.pca_comp_edit.text()),
            center=True,
            scale=False,
        )
        pre = HSIPreprocessor(pre_cfg)
        feat = HSIFeatureExtractor(feat_cfg)
        return pre, feat

    def _on_show_raw_spectra(self) -> None:
        try:
            spectra, labels, _, wavelengths, label_names = self._load_sample_spectra(
                int(self.max_spectra_edit.text()))
            fig = Figure(figsize=(8.5, 4.5))
            ax = fig.add_subplot(111)
            wl = np.asarray(wavelengths) if wavelengths is not None else np.arange(
                spectra.shape[1])
            labels_u = np.unique(labels)
            cmap = _build_class_cmap(labels_u)
            try:
                color_list = list(cmap.colors)
            except Exception:
                color_list = [cmap(i) for i in range(len(labels_u))]
            label_to_idx = {int(lbl): idx for idx, lbl in enumerate(labels_u)}
            for lbl in labels_u:
                mask = labels == lbl
                if not np.any(mask):
                    continue
                xs = wl if wl.size else np.arange(spectra.shape[1])
                color = color_list[label_to_idx[int(lbl)]]
                for row in spectra[mask][:200]:
                    ax.plot(xs, row, color=color, alpha=0.08, linewidth=0.8)
                mean = np.mean(spectra[mask], axis=0)
                ax.plot(xs, mean, color=color, linewidth=1.2,
                        label=_format_class_label(int(lbl), label_names))
            ax.set_title("Spectra: Sampled")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Reflectance")
            ax.legend(loc="upper right", fontsize=8)
            self._show_figure_popup(fig)
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Raw spectra plot failed:\n{exc}")

    def _on_show_mean_std(self) -> None:
        try:
            spectra, labels, _, wavelengths, label_names = self._load_sample_spectra(
                int(self.max_spectra_edit.text()))
            labels_u = np.unique(labels)
            wl = np.asarray(wavelengths) if wavelengths is not None else np.arange(
                spectra.shape[1])
            fig = Figure(figsize=(8.5, 4.5))
            ax = fig.add_subplot(111)
            cmap = _build_class_cmap(labels_u)
            try:
                color_list = list(cmap.colors)
            except Exception:
                color_list = [cmap(i) for i in range(len(labels_u))]
            for i, lbl in enumerate(labels_u):
                mask = labels == lbl
                if not np.any(mask):
                    continue
                data = spectra[mask].astype(np.float32)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                color = color_list[i]
                ax.plot(wl, mean, color=color, linewidth=1.2,
                        label=_format_class_label(int(lbl), label_names))
                ax.fill_between(wl, mean - std, mean + std,
                                color=color, alpha=0.22)
            ax.set_title("Spectra: Mean ± Std")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Reflectance")
            ax.legend(loc="upper right", fontsize=8)
            self._show_figure_popup(fig)
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Mean±Std plot failed:\n{exc}")

    def _on_show_dataset_preview(self) -> None:
        try:
            dataset = self.dataset_edit.text().strip()
            if not dataset:
                raise ValueError("Please set dataset path")

            sample_name = self.preview_sample_combo.currentData()
            label_names, _ = _load_label_names(dataset)
            sample_name, sample_rgb, gt_map, _ = _prepare_sample_preview_data(
                dataset_path=dataset,
                preprocessor=None,
                feature_extractor=None,
                sample_name=sample_name,
            )

            fig = Figure(figsize=(8.5, 6.0))
            ax = fig.add_subplot(111)
            if sample_rgb.dtype != np.float32 and sample_rgb.max() > 1:
                rgb = sample_rgb.astype(np.float32)
                if rgb.max() > 0:
                    rgb /= rgb.max()
            else:
                rgb = sample_rgb.astype(np.float32)

            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(f"Dataset Preview: {sample_name}")

            labels_u = np.unique(gt_map)
            labels_u = labels_u[labels_u != 0] if labels_u.size else labels_u
            cmap = _build_class_cmap(labels_u)
            try:
                color_list = list(cmap.colors)
            except Exception:
                color_list = [cmap(i) for i in range(len(labels_u))]

            ann_samples = _collect_sample_annotations_from_json(
                dataset, sample_name)
            legend_entries = []
            self._preview_ann_info = []

            if ann_samples:
                # render each annotation (per-object) with id and class label
                for mask_arr, ann_class, ann_id in ann_samples:
                    ann_cnts, _ = cv2.findContours(mask_arr.astype(
                        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if ann_class in labels_u.tolist():
                        i = int(np.where(labels_u == ann_class)[0][0])
                        color = color_list[i]
                    else:
                        color = "red"

                    class_label = _format_class_label(
                        int(ann_class), label_names)
                    paths = []
                    for c in ann_cnts:
                        contour = c.reshape(-1, 2)
                        if not np.array_equal(contour[0], contour[-1]):
                            contour = np.vstack([contour, contour[0]])
                        paths.append(mpath.Path(contour))
                        ax.plot(contour[:, 0], contour[:, 1],
                                color=color, linewidth=1.4)
                        if contour.size > 0:
                            cx = int(np.mean(contour[:, 0]))
                            cy = int(np.mean(contour[:, 1]))
                            ax.text(cx, cy, f"{class_label}:{int(ann_id)}", color="white", fontsize=7,
                                    bbox=dict(facecolor="black", alpha=0.6, pad=1))

                    self._preview_ann_info.append({
                        'ann_id': int(ann_id),
                        'class_id': int(ann_class),
                        'paths': paths,
                        'mask': mask_arr,
                        'color': color,
                    })

                # legend by class, not annotation id
                for idx, lbl in enumerate(labels_u):
                    legend_entries.append(Patch(
                        facecolor=color_list[idx], edgecolor=color_list[idx], label=_format_class_label(int(lbl), label_names)))
            else:
                for i, lbl in enumerate(labels_u):
                    mask = (gt_map == lbl).astype(np.uint8)
                    if not np.any(mask):
                        continue
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color = color_list[i]
                    for contour in contours:
                        contour = contour.reshape(-1, 2)
                        if not np.array_equal(contour[0], contour[-1]):
                            contour = np.vstack([contour, contour[0]])
                        ax.plot(contour[:, 0], contour[:, 1],
                                color=color, linewidth=1.2)

                    legend_entries.append(
                        Patch(facecolor=color, edgecolor=color, label=_format_class_label(int(lbl), label_names)))

                try:
                    moments = cv2.moments(mask)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        ax.text(cx, cy, _format_class_label(int(lbl), label_names), color="white", fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.5, pad=1))
                except Exception:
                    pass

            if legend_entries:
                ax.legend(handles=legend_entries, loc="upper right",
                          fontsize=8, title="GT classes")

            self._preview_hsi_cube = None
            hsi_hdr = _resolve_hsi_header_path(
                os.path.join(dataset, 'HSI'), sample_name)
            if os.path.exists(hsi_hdr):
                cube = spy.open_image(hsi_hdr)
                self._preview_hsi_cube = np.asarray(
                    cube.open_memmap(), dtype=np.float32)

            # Own modal popup with event bindings
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Preview - {sample_name}")
            dlg.resize(900, 700)
            layout = QVBoxLayout(dlg)
            canvas = FigureCanvasQTAgg(fig)
            canvas.mpl_connect(
                'motion_notify_event', lambda event: self._on_preview_hover(event, ax, dlg))
            canvas.mpl_connect('button_press_event', lambda event: self._on_preview_click(
                event, dataset, sample_name))
            layout.addWidget(canvas)
            dlg.exec_()

        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Dataset preview failed:\n{exc}")

    def _on_preview_hover(self, event, ax, dlg):
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        selected_ann = None
        # first check inside mask at pixel coordinate (robust for edge areas)
        xi = int(round(x))
        yi = int(round(y))
        for ann in self._preview_ann_info:
            mask = ann.get('mask')
            if mask is not None:
                h, w = mask.shape[:2]
                if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
                    selected_ann = ann
                    break

        # fallback: path geometry (requires paths created from contours)
        if selected_ann is None:
            for ann in self._preview_ann_info:
                for path in ann['paths']:
                    if path.contains_point((x, y), radius=1.0):
                        selected_ann = ann
                        break
                if selected_ann is not None:
                    break

        if selected_ann is None:
            self._preview_current_ann_id = None
            if self._preview_hover_patch is not None:
                self._preview_hover_patch.remove()
                self._preview_hover_patch = None
            if self._preview_hover_text is not None:
                self._preview_hover_text.remove()
                self._preview_hover_text = None
            dlg.repaint()
            return

        if self._preview_current_ann_id == selected_ann['ann_id']:
            return

        self._preview_current_ann_id = selected_ann['ann_id']

        # remove previous hover patch and text
        if self._preview_hover_patch is not None:
            self._preview_hover_patch.remove()
            self._preview_hover_patch = None
        if self._preview_hover_text is not None:
            self._preview_hover_text.remove()
            self._preview_hover_text = None

        # draw filled mask region for hover (interior fill, not just edge)
        mask_img = np.zeros((*selected_ann['mask'].shape, 4), dtype=np.float32)
        mask_bool = selected_ann['mask'].astype(bool)
        mask_img[..., 0] = 1.0  # red channel
        mask_img[..., 1] = 0.7  # green channel
        mask_img[..., 2] = 0.0  # blue channel
        mask_img[..., 3] = np.where(mask_bool, 0.25, 0.0)

        if self._preview_hover_patch is not None:
            self._preview_hover_patch.remove()
            self._preview_hover_patch = None
        if self._preview_hover_im is not None:
            self._preview_hover_im.remove()
            self._preview_hover_im = None

        # use the class color for fill, with alpha
        fill_color = selected_ann.get('color', (1, 1, 0))
        try:
            rgba = mcolors.to_rgba(fill_color)
        except Exception:
            rgba = (1.0, 1.0, 0.0, 1.0)

        mask_color = np.zeros(
            (*selected_ann['mask'].shape, 4), dtype=np.float32)
        mask_color[..., 0] = rgba[0]
        mask_color[..., 1] = rgba[1]
        mask_color[..., 2] = rgba[2]
        mask_color[..., 3] = np.where(mask_bool, 0.35, 0.0)

        self._preview_hover_im = ax.imshow(
            mask_color, interpolation='none', origin='upper')
        # force redraw on canvas so hover feedback appears immediately
        try:
            event.canvas.draw_idle()
        except Exception:
            pass
        dlg.repaint()

    def _on_preview_click(self, event, dataset, sample_name):
        if event.dblclick and event.button == 1:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return

            selected_ann = None
            for ann in self._preview_ann_info:
                for path in ann['paths']:
                    if path.contains_point((x, y)):
                        selected_ann = ann
                        break
                if selected_ann is not None:
                    break

            if selected_ann is not None:
                self._show_annotation_spectra(
                    dataset, sample_name, selected_ann)

    def _show_annotation_spectra(self, dataset, sample_name, ann_info):
        if self._preview_hsi_cube is None:
            hsi_hdr = _resolve_hsi_header_path(
                os.path.join(dataset, 'HSI'), sample_name)
            if not os.path.exists(hsi_hdr):
                QMessageBox.warning(
                    self, 'No HSI', 'HSI file not found for this sample')
                return
            cube = spy.open_image(hsi_hdr)
            self._preview_hsi_cube = np.asarray(
                cube.open_memmap(), dtype=np.float32)

        mask = ann_info['mask'].astype(bool)
        coords = np.argwhere(mask)
        if coords.size == 0:
            QMessageBox.information(
                self, 'No annotation pixels', 'No pixels found for annotation')
            return

        spec = self._preview_hsi_cube[coords[:, 0], coords[:, 1], :]

        # Plot maximum 50 random spectra for visual clarity, but compute stats from the same plotted subset to avoid mean outside visible range.
        n_pixels = spec.shape[0]
        max_display = min(n_pixels, 50)
        if n_pixels > max_display:
            rng = np.random.default_rng(42)
            display_idx = rng.choice(n_pixels, size=max_display, replace=False)
        else:
            display_idx = np.arange(n_pixels)

        display_spec = spec[display_idx]
        mean_spec = np.mean(display_spec, axis=0)
        std_spec = np.std(display_spec, axis=0)
        min_spec = np.min(display_spec, axis=0)
        max_spec = np.max(display_spec, axis=0)

        fig = Figure(figsize=(8.5, 4.5))
        ax = fig.add_subplot(111)
        for i in range(display_spec.shape[0]):
            ax.plot(display_spec[i], color='gray', alpha=0.18)

        ann_color = ann_info.get('color', 'red')
        try:
            ann_rgba = mcolors.to_rgba(ann_color)
        except Exception:
            ann_rgba = (1.0, 0.0, 0.0, 1.0)

        ax.fill_between(np.arange(
            mean_spec.shape[0]), min_spec, max_spec, color=ann_rgba, alpha=0.1, label='min-max')
        ax.plot(mean_spec, color=ann_rgba, linewidth=2, label='mean')
        ax.fill_between(np.arange(mean_spec.shape[0]), mean_spec - std_spec, mean_spec + std_spec,
                        color=ann_rgba, alpha=0.25, label='±std')

        ax.set_title(
            f"Annotation {ann_info['ann_id']} (class {ann_info['class_id']}) spectra"
        )
        ax.set_xlabel('Band index')
        ax.set_ylabel('Reflectance')
        ax.legend()
        self._show_figure_popup(fig)

    def _on_show_2d_feature_viz(self) -> None:
        try:
            spectra, labels, _, wavelengths, label_names = self._load_sample_spectra(
                int(self.max_spectra_edit.text()))
            pre, feat = self._build_preprocessor_and_extractor()
            X_pre = pre.fit_transform(spectra, wavelengths=wavelengths)
            X_feat = feat.fit_transform(X_pre)
            centered = X_feat - np.mean(X_feat, axis=0, keepdims=True)
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                comp = vt[:2].T
                xy = centered @ comp
            except np.linalg.LinAlgError:
                xy = centered[:, :2]

            fig = Figure(figsize=(7.5, 6.0))
            ax = fig.add_subplot(111)
            labels_u = np.unique(labels)
            cmap = _build_class_cmap(labels_u)
            try:
                color_list = list(cmap.colors)
            except Exception:
                color_list = [cmap(i) for i in range(len(labels_u))]
            label_to_idx = {int(lbl): idx for idx, lbl in enumerate(labels_u)}
            for lbl in labels_u:
                idxs = labels == lbl
                if not np.any(idxs):
                    continue
                color = color_list[label_to_idx[int(lbl)]]
                ax.scatter(xy[idxs, 0], xy[idxs, 1], s=8, c=[
                           color], alpha=0.6, label=_format_class_label(int(lbl), label_names))
            ax.set_title("Feature 2D Visualization")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(loc="best")
            self._show_figure_popup(fig)
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"2D feature viz failed:\n{exc}")

    def _on_show_3d_feature_viz(self) -> None:
        try:
            spectra, labels, _, wavelengths, label_names = self._load_sample_spectra(
                int(self.max_spectra_edit.text()))
            pre, feat = self._build_preprocessor_and_extractor()
            X_pre = pre.fit_transform(spectra, wavelengths=wavelengths)
            X_feat = feat.fit_transform(X_pre)
            centered = X_feat - np.mean(X_feat, axis=0, keepdims=True)
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                comp = vt[:3].T
                xyz = centered @ comp
            except np.linalg.LinAlgError:
                if centered.shape[1] >= 3:
                    xyz = centered[:, :3]
                else:
                    pad = np.zeros(
                        (centered.shape[0], max(0, 3 - centered.shape[1])))
                    xyz = np.hstack([centered, pad])

            fig = Figure(figsize=(7.5, 6.0))
            ax = fig.add_subplot(111, projection="3d")
            labels_u = np.unique(labels)
            cmap = _build_class_cmap(labels_u)
            try:
                color_list = list(cmap.colors)
            except Exception:
                color_list = [cmap(i) for i in range(len(labels_u))]
            label_to_idx = {int(lbl): idx for idx, lbl in enumerate(labels_u)}
            for lbl in labels_u:
                idxs = labels == lbl
                if not np.any(idxs):
                    continue
                color = color_list[label_to_idx[int(lbl)]]
                ax.scatter(xyz[idxs, 0], xyz[idxs, 1], xyz[idxs, 2], s=6, color=color,
                           alpha=0.6, label=_format_class_label(int(lbl), label_names))
            ax.set_title("PCA - 3D Visualization")
            ax.set_xlabel("PC-1")
            ax.set_ylabel("PC-2")
            ax.set_zlabel("PC-3")
            ax.legend(loc="best")
            self._show_figure_popup(fig)
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"3D feature viz failed:\n{exc}")

    def _on_show_normalize_plot(self) -> None:
        try:
            spectra, labels, _, wavelengths, _lnames = self._load_sample_spectra(
                int(self.max_spectra_edit.text()))
            pre, _ = self._build_preprocessor_and_extractor()
            wl = np.asarray(wavelengths) if wavelengths is not None else np.arange(
                spectra.shape[1])
            idxs = np.random.default_rng(42).choice(
                np.arange(spectra.shape[0]), size=min(50, spectra.shape[0]), replace=False)
            raw_sample = spectra[idxs].astype(np.float32)
            pre.fit_transform(raw_sample, wavelengths=wl)
            transformed = pre.transform(raw_sample)

            fig = Figure(figsize=(8.5, 4.5))
            ax = fig.add_subplot(111)
            for i in range(raw_sample.shape[0]):
                ax.plot(wl, raw_sample[i], color="#999999", alpha=0.12)
            for i in range(transformed.shape[0]):
                ax.plot(wl, transformed[i], color="#2a7fff", alpha=0.18)
            ax.set_title("Normalize: raw (grey) vs normalized (blue)")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Value")
            self._show_figure_popup(fig)
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Normalize plot failed:\n{exc}")

    def _on_show_global_scale_plot(self) -> None:
        try:
            spectra, labels, _, wavelengths, _lnames = self._load_sample_spectra(
                int(self.max_spectra_edit.text()))
            pre, _ = self._build_preprocessor_and_extractor()
            wl = np.asarray(wavelengths) if wavelengths is not None else np.arange(
                spectra.shape[1])
            raw = spectra.astype(np.float32)
            transformed = pre.fit_transform(raw, wavelengths=wl)
            mean_raw = np.mean(raw, axis=0)
            std_raw = np.std(raw, axis=0)
            mean_tr = np.mean(transformed, axis=0)
            std_tr = np.std(transformed, axis=0)

            fig = Figure(figsize=(9.0, 4.6))
            ax1 = fig.add_subplot(121)
            ax1.plot(wl, mean_raw, label="mean raw")
            ax1.plot(wl, mean_tr, label="mean transformed")
            ax1.set_title("Per-band Mean: raw vs transformed")
            ax1.set_xlabel("Wavelength (nm)")
            ax1.legend(fontsize=7)

            ax2 = fig.add_subplot(122)
            ax2.plot(wl, std_raw, label="std raw")
            ax2.plot(wl, std_tr, label="std transformed")
            ax2.set_title("Per-band Std: raw vs transformed")
            ax2.set_xlabel("Wavelength (nm)")
            ax2.legend(fontsize=7)

            self._show_figure_popup(fig)
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Global scale plot failed:\n{exc}")

    def _build_summary_figure(
        self,
        sample_name: str,
        sample_rgb: np.ndarray,
        pred_map: np.ndarray,
        preview_metrics: dict[str, object],
        best_model_name: str,
        class_labels: np.ndarray,
        label_names: dict[int, str],
        page_label: str = "Preview",
        annotation_mask: Optional[np.ndarray] = None,
    ) -> Figure:
        fig = Figure(figsize=(8.6, 7.4), constrained_layout=False)
        # Two rows: top for sample+prediction, bottom for confusion matrix.
        # Legend will be placed as a figure-level legend below the axes.
        gs = fig.add_gridspec(2, 2, height_ratios=[
                              1.0, 1.12], hspace=0.18, wspace=0.12)
        fig.suptitle(f"{page_label} Split - {best_model_name}",
                     fontsize=9, fontweight="bold", y=0.988)
        # fig.text(
        #     0.5,
        #     0.932,
        #     f"Sample page: {sample_name}",
        #     ha="center",
        #     va="top",
        #     fontsize=8.2,
        #     color="#444444",
        # )

        # mask prediction to selected sample (or split subset) to avoid showing outside pixels.
        pred_map_plot = np.asarray(pred_map, dtype=np.int32).copy()
        if annotation_mask is not None:
            mask = np.asarray(annotation_mask, dtype=bool)
            if mask.shape == pred_map_plot.shape:
                pred_map_plot[~mask] = 0
        else:
            # for summary and pdf when no explicit split mask is provided,
            # keep only predicted (nonzero) pixels to avoid highlighting outside sample.
            pred_map_plot[pred_map_plot == 0] = 0

        # แทนที่ block ตั้งแต่ present_values ถึง norm

        present_values = np.unique(pred_map_plot.reshape(-1))
        present_labels = np.asarray(
            [int(v) for v in present_values if int(v) != 0], dtype=np.int32)

        if present_labels.size > 0:
            ordered_present = _build_metric_label_order(class_labels, present_labels)
            class_labels_present = np.asarray(
                [lbl for lbl in ordered_present.tolist() if int(lbl) in present_labels],
                dtype=np.int32,
            )
        else:
            class_labels_present = np.array([], dtype=np.int32)

        # รวม 0 (background/outside) ไว้ที่ index 0 เสมอ
        all_labels = np.concatenate([[0], class_labels_present]).astype(np.int32)
        display_labels = class_labels_present  # legend ไม่แสดง 0

        label_to_idx = {int(lbl): idx for idx, lbl in enumerate(all_labels)}
        pred_idx = np.zeros_like(pred_map_plot, dtype=np.int32)  # default = 0 = black
        for lbl, idx in label_to_idx.items():
            pred_idx[pred_map_plot == lbl] = idx

        # สร้าง cmap โดยให้ index 0 = ดำ, ที่เหลือเป็นสีตาม class
        jet = colormaps.get_cmap("jet")
        if class_labels_present.size == 0:
            class_colors = []
        elif class_labels_present.size == 1:
            class_colors = [jet(0.5)]
        else:
            class_colors = [jet(v) for v in np.linspace(0.05, 0.95, class_labels_present.size)]

        color_list = [(0.0, 0.0, 0.0, 1.0)] + class_colors  # index 0 = ดำ
        cmap = mcolors.ListedColormap(color_list)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(all_labels) + 0.5, 1), cmap.N)
        legend_labels = [int(lbl) for lbl in display_labels.tolist()]

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(sample_rgb)
        ax1.set_title(f"{page_label} Sample", fontsize=10)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(sample_rgb, alpha=0.35)
        ax2.imshow(pred_idx, cmap=cmap, norm=norm,
                   alpha=0.65, interpolation="nearest")
        ax2.set_title(
            f"Prediction Map ({best_model_name} | {page_label.lower()} sample)",
            fontsize=11,
        )
        ax2.axis("off")
        # Build legend handles (figure-level) so legend is fully outside axes
        legend_handles = []
        if legend_labels:
            legend_handles = [
                Patch(facecolor=cmap(label_to_idx[int(
                    lbl)]), edgecolor="none", label=_format_class_name(int(lbl), label_names))
                for lbl in legend_labels
            ]
            # Choose number of columns to keep legend compact and predictable
            ncol = len(legend_handles)

            # Place legend at figure level, below all subplots
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=ncol,
                fontsize=6.0,
                frameon=True,
                framealpha=0.92,
                title="Classes",
                title_fontsize=7,
            )

        ax3 = fig.add_subplot(gs[1, :])
        self._draw_confusion_on_axis(
            fig, ax3,
            f"{page_label} Confusion Matrix - {best_model_name}",
            preview_metrics, label_names,
            apply_subplots_adjust=False   # ✅ default แล้ว ไม่ต้องส่งก็ได้
        )
        # Adjust margins to accommodate figure-level legend and colorbar
        fig.subplots_adjust(left=0.06, right=0.88, top=0.94, bottom=0.17)
        return fig

    # _build_confusion_figure → standalone figure → ต้องการ adjust
    def _build_confusion_figure(self, model_name, metrics, label_names):
        fig = Figure(figsize=(5.9, 5.0), constrained_layout=True)
        ax = fig.add_subplot(111)
        self._draw_confusion_on_axis(
            fig, ax, model_name, metrics, label_names,
            apply_subplots_adjust=True   # ✅
        )
        return fig

    def _draw_confusion_on_axis(
        self,
        fig: Figure,
        ax,
        model_name: str,
        metrics: dict[str, object],
        label_names: dict[int, str],
        *,
        apply_subplots_adjust: bool = False,   # ← เพิ่ม flag นี้
    ) -> None:
        cm = np.asarray(metrics["confusion_matrix"], dtype=np.float32)
        labels = np.asarray(metrics["labels"], dtype=np.int32)

        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        active_idx = np.array(
            [i for i in range(len(labels)) if row_sums[i] > 0 or col_sums[i] > 0],
            dtype=np.int64,
        )
        if active_idx.size > 0 and active_idx.size < len(labels):
            cm = cm[np.ix_(active_idx, active_idx)]
            labels = labels[active_idx]

        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        cm_norm = cm / denom

        im = ax.imshow(cm_norm, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)

        # ✅ แยก label_strs ออก ไม่ overwrite labels (int array)
        label_strs = [_format_class_name(int(x), label_names) for x in labels]
        ax.set_xticks(np.arange(len(label_strs)))
        ax.set_yticks(np.arange(len(label_strs)))
        ax.set_xticklabels(label_strs, rotation=0, ha="center", fontsize=7)
        ax.set_yticklabels(label_strs, rotation=90, va="center", fontsize=7)
        ax.tick_params(axis="x", pad=6)
        ax.tick_params(axis="y", pad=4)
        ax.set_title(f"{model_name} - Confusion Matrix (Normalized)", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

        # ✅ เรียก subplots_adjust เฉพาะตอนวาด standalone (confusion-only figure)
        if apply_subplots_adjust:
            fig.subplots_adjust(left=0.24, right=0.92, bottom=0.28, top=0.90)

        threshold = cm_norm.max() / 2.0 if cm_norm.size else 0.0
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                count = int(cm[i, j])
                proportion = float(cm_norm[i, j])
                ax.text(
                    j, i,
                    f"{count}\n{proportion:.2f}",
                    ha="center", va="center",
                    color="white" if proportion > threshold else "black",
                    fontsize=7,
                )

    def _export_report_pdf(self) -> None:
        if self._last_result is None:
            QMessageBox.information(
                self, "No Results", "Please run models before exporting report.")
            return

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"hsi_report_{now}.pdf"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report PDF",
            os.path.join(self.dataset_edit.text().strip()
                         or os.getcwd(), default_name),
            "PDF Files (*.pdf)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".pdf"):
            out_path += ".pdf"

        result = self._last_result
        cfg = self._last_run_config
        best_name = str(result["best_model_name"])
        best_labels = self._get_best_labels(result, best_name)
        label_names = result.get("label_names", {})

        try:
            with PdfPages(out_path) as pdf:
                cover = self._build_report_cover_figure(result, cfg)
                self._apply_pdf_page_margins(
                    cover, left=0.08, right=0.94, top=0.95, bottom=0.08)
                pdf.savefig(cover)

                source_table = self._build_source_split_table_figure(
                    result, cfg)
                self._apply_pdf_page_margins(
                    source_table, left=0.08, right=0.94, top=0.93, bottom=0.09)
                pdf.savefig(source_table)

                diagram_page = self._build_methodology_diagram(cfg, result)
                pdf.savefig(diagram_page)

                split_figure = self._build_train_test_figure(result)
                self._apply_pdf_page_margins(
                    split_figure, left=0.07, right=0.95, top=0.93, bottom=0.09)
                pdf.savefig(split_figure)

                summary_text = Figure(
                    figsize=(8.27, 11.69), constrained_layout=True)
                ax_text = summary_text.add_subplot(111)
                ax_text.axis("off")
                ax_text.text(
                    0.06,
                    0.95,
                    "\n".join(self._build_summary_lines(
                        result, best_labels, label_names)),
                    va="top",
                    ha="left",
                    fontsize=9,
                    family="monospace",
                )
                self._apply_pdf_page_margins(
                    summary_text, left=0.08, right=0.94, top=0.95, bottom=0.08)
                pdf.savefig(summary_text)

                split_reports = result.get("split_reports", {})
                for split_key in ("train", "validation", "test"):
                    report_list = split_reports.get(split_key, [])
                    # Support both old single-dict format and new list format
                    if isinstance(report_list, dict):
                        report_list = [report_list] if report_list else []
                    for report in report_list:
                        if not report:
                            continue
                        split_page = self._build_summary_figure(
                            sample_name=str(report["sample_name"]),
                            sample_rgb=np.asarray(report["sample_rgb"]),
                            pred_map=np.asarray(report["pred_map"]),
                            preview_metrics=dict(report["metrics"]),
                            best_model_name=str(report["best_model_name"]),
                            class_labels=best_labels,
                            label_names=label_names,
                            page_label=str(report["page_label"]),
                            annotation_mask=np.asarray(
                                report.get("gt_map", np.zeros_like(
                                    report.get("pred_map", np.zeros(
                                        (1, 1), dtype=np.int32))
                                )) != 0
                            ),
                        )
                        # Increase bottom margin to provide space for the legend placed
                        # under the prediction map so it doesn't overlap the confusion matrix.
                        self._apply_pdf_page_margins(
                            split_page, left=0.08, right=0.94, top=0.93, bottom=0.20)
                        pdf.savefig(split_page)

                _fi_names = result.get("feature_names", [])
                for item in result["model_results"]:
                    _fi = item.get("feature_importances")
                    _has_fi = _fi is not None and len(_fi) > 0

                    # Page 1 (always): metrics + confusion matrix
                    page = Figure(figsize=(11.69, 8.27),
                                  constrained_layout=False)
                    gs = page.add_gridspec(1, 2, width_ratios=[
                                           0.95, 1.35], wspace=0.22)

                    ax_left = page.add_subplot(gs[0, 0])
                    ax_left.axis("off")
                    ax_left.text(
                        0.03,
                        0.97,
                        self._format_model_metrics_text(
                            str(item["name"]), item["metrics"], label_names),
                        va="top",
                        ha="left",
                        fontsize=8.5,
                        family="monospace",
                    )

                    ax_cm = page.add_subplot(gs[0, 1])
                    self._draw_confusion_on_axis(
                        page, ax_cm, str(item["name"]), item["metrics"], label_names,
                        apply_subplots_adjust=True   # ✅
                    )
                    self._apply_pdf_page_margins(
                        page, left=0.05, right=0.97, top=0.93, bottom=0.09)
                    pdf.savefig(page)

                    if _has_fi:
                        # Page 2 (optional): full-page feature importance
                        fi_page = Figure(figsize=(11.69, 8.27),
                                         constrained_layout=False)
                        fi_ax = fi_page.add_subplot(111)
                        self._draw_feature_importance_on_axis(
                            fi_ax,
                            np.asarray(_fi),
                            list(_fi_names),
                            str(item["name"]),
                        )
                        self._apply_pdf_page_margins(
                            fi_page, left=0.11, right=0.96, top=0.92, bottom=0.10)
                        pdf.savefig(fi_page)

            QMessageBox.information(
                self, "Export Complete", f"PDF report saved:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed",
                                 f"Failed to export PDF report:\n{exc}")

    def _build_report_cover_figure(self, result: dict[str, object], cfg: Optional[PipelineInput]) -> Figure:
        fig = Figure(figsize=(8.27, 11.69), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.axis("off")

        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "HSI Classification Report",
            "=" * 70,
            f"Generated at: {run_time}",
            "",
            "Pipeline Settings",
            "-" * 70,
        ]

        if cfg is not None:
            lines.extend(
                [
                    f"Dataset path        : {cfg.dataset_path}",
                    f"Page order          : Train / Validation / Test",
                    f"Max spectra/sample  : {cfg.max_spectra}",
                    f"Validation ratio    : {cfg.val_ratio}",
                    f"Train/Test weight   : {cfg.test_ratio}",
                    f"Spectral norm       : {cfg.spectral_norm}",
                    f"Global scaling      : {cfg.global_scale}",
                    f"Feature mode        : {cfg.feature_mode}",
                    f"PCA components      : {cfg.pca_components}",
                    f"k neighbors         : {cfg.k_neighbors}",
                    f"Selected models ({len(cfg.selected_models)}):",
                    "  " + ", ".join(cfg.selected_models),
                ]
            )

        lines.extend(
            [
                "",
                "Run Summary",
                "-" * 70,
                f"Train samples       : {result['n_train']}",
                f"Validation samples  : {result['n_val']}",
                f"Test samples        : {result.get('n_test', 0)}",
                f"Feature dimension   : {result['feature_dim']}",
                f"Best model          : {result['best_model_name']}",
                f"Successful models   : {len(result['model_results'])}",
                f"Failed models       : {len(result.get('failed_models', []))}",
                f"Label names file    : {'loaded' if result.get('label_names') else 'not found'}",
                "",
                "This report contains:",
                "1. Dataset source summary table (train/val/test split)",
                "2. Methodology Diagram",
                "3. Train/Test plots (class distribution + 2D split map)",
                "4. Summary metrics and selected classes",
                "5. Train / Validation / Test sample pages",
                "6. Per-model metrics, confusion matrix, and feature importance",
            ]
        )

        ax.text(0.06, 0.95, "\n".join(lines), va="top",
                ha="left", fontsize=9, family="monospace")
        return fig

    def _build_methods_figure(self, cfg: Optional[PipelineInput]) -> Figure:
        fig = Figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")

        spectral_norm = cfg.spectral_norm if cfg else "N/A"
        global_scale = cfg.global_scale if cfg else "N/A"
        feature_mode = cfg.feature_mode if cfg else "N/A"
        pca_comp = str(
            cfg.pca_components) if cfg and cfg.pca_components is not None else "None"

        spectral_norm_desc = {
            "snv":  "Standard Normal Variate (SNV) — subtracts the spectrum mean and divides by\n"
                    "        its standard deviation: x̂ = (x − μ) / σ. SNV removes multiplicative\n"
                    "        scatter and baseline effects that arise from differences in particle\n"
                    "        size and path length (Barnes et al., 1989).",
            "l2":   "L2 (Euclidean) normalisation — divides each spectrum by its L2 norm:\n"
                    "        x̂ = x / ‖x‖₂, projecting all spectra onto the unit hypersphere.",
            "max":  "Max normalisation — divides each spectrum by its maximum absolute value:\n"
                    "        x̂ = x / max|x|, scaling intensities to [−1, 1].",
            "area": "Area normalisation — divides by the sum of absolute band values:\n"
                    "        x̂ = x / Σ|xᵢ|, making total spectral area equal to unity.",
            "none": "No spectral normalisation applied.",
        }.get(spectral_norm, spectral_norm)

        global_scale_desc = {
            "standard": "Standardisation (Z-score) — for each band b, mean μ_b and standard\n"
                        "        deviation σ_b are estimated from training spectra only and then applied\n"
                        "        at inference: x̂_b = (x_b − μ_b) / σ_b.",
            "minmax":   "Min–Max scaling — maps each band into [0, 1] using train-set extrema:\n"
                        "        x̂_b = (x_b − min_b) / (max_b − min_b).",
            "none":     "No global scaling applied.",
        }.get(global_scale, global_scale)

        feature_mode_desc = {
            "raw":      "Raw spectra — the preprocessed spectrum vector x ∈ ℝᴮ (B = number of bands)\n"
                        "        is used directly as the feature vector.",
            "stat":     "Spectral statistics (hand-crafted) — a 10-dimensional descriptor is computed\n"
                        "        per spectrum: {mean, std, min, max, peak-to-peak range, spectral area,\n"
                        "        L2 norm, mean absolute value, end-to-start slope, skewness-like moment}.\n"
                        "        This compact representation discards redundant inter-band correlation\n"
                        "        while retaining shape-level discriminative cues.",
            "pca":      "Principal Component Analysis (PCA) — the centred and optionally scaled\n"
                        "        spectra are projected onto the leading eigenvectors of the training\n"
                        "        covariance matrix via truncated SVD. The number of components is\n"
                        f"        determined by a retained-variance threshold of {pca_comp}\n"
                        "        (Jolliffe, 2002).",
            "pca_stat": "PCA + Statistics (concatenated) — the PCA projection (see above) and the\n"
                        "        10-dimensional statistical descriptor are concatenated, yielding a hybrid\n"
                        "        feature that combines global variance structure with local spectral shape.",
        }.get(feature_mode, feature_mode)

        sections = [
            ("2. Methods", None, True, 14),
            ("", None, False, 9),
            ("2.1 Spectral Preprocessing", None, True, 11),
            (
                "Preprocessing transforms raw detector counts into normalised, scale-invariant\n"
                "representations suitable for machine-learning classifiers. The pipeline applies\n"
                "the following steps in order; all statistics are estimated exclusively from the\n"
                "training partition and re-applied at inference to prevent data leakage.",
                None, False, 9,
            ),
            ("", None, False, 9),
            ("Step 1 — Band removal", None, True, 9.5),
            (
                "Spectral bands falling within user-specified wavelength ranges (e.g. water-\n"
                "absorption regions ~1400 nm and ~1900 nm) are discarded prior to any\n"
                "normalisation.",
                None, False, 9,
            ),
            ("", None, False, 9),
            ("Step 2 — Per-spectrum normalisation", None, True, 9.5),
            (f"Method selected: {spectral_norm}", None, False, 9),
            (spectral_norm_desc, None, False, 9),
            ("", None, False, 9),
            ("Step 3 — Global feature scaling", None, True, 9.5),
            (f"Method selected: {global_scale}", None, False, 9),
            (global_scale_desc, None, False, 9),
            ("", None, False, 9),
            ("2.2 Feature Engineering", None, True, 11),
            (
                "Feature engineering maps the preprocessed B-band spectrum of each pixel into a\n"
                "compact, classifier-friendly representation. The extractor is fitted on the\n"
                "training partition only.",
                None, False, 9,
            ),
            ("", None, False, 9),
            (f"Method selected: {feature_mode}", None, True, 9.5),
            (feature_mode_desc, None, False, 9),
            ("", None, False, 9),
            ("References", None, True, 9.5),
            (
                "Barnes, R.J., Dhanoa, M.S., & Lister, S.J. (1989). Standard normal variate\n"
                "  transformation and de-trending of near-infrared diffuse reflectance spectra.\n"
                "  Applied Spectroscopy, 43(5), 772–777.\n"
                "Jolliffe, I.T. (2002). Principal Component Analysis (2nd ed.). Springer.",
                None, False, 8.5,
            ),
        ]

        y = 0.97
        for text, _, bold, fsize in sections:
            ax.text(
                0.03, y, text,
                ha="left", va="top",
                fontsize=fsize,
                fontweight="bold" if bold else "normal",
                family="serif",
                transform=ax.transAxes,
                wrap=False,
                linespacing=1.45,
            )
            lines = text.count("\n") + 1 if text.strip() else 0.5
            y -= (fsize / 72.0) * 1.55 * lines + 0.005
            if y < 0.02:
                break

        return fig

    def _build_methodology_diagram(self, cfg: Optional[PipelineInput], result: dict[str, object]) -> Figure:
        """
        Render a horizontal pipeline diagram:
        Raw HSI Data → Preprocessing → Feature Engineering → Model Training → Evaluation
        Each box shows the actual settings used.
        """
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        fig = Figure(figsize=(11.69, 5.5))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        fig.patch.set_facecolor("#f8f9fb")

        spectral_norm = cfg.spectral_norm if cfg else "snv"
        global_scale = cfg.global_scale if cfg else "standard"
        feature_mode = cfg.feature_mode if cfg else "stat"
        pca_comp = str(
            cfg.pca_components) if cfg and cfg.pca_components is not None else "auto"
        n_models = len(result.get("model_results", []))
        best_model = str(result.get("best_model_name", "N/A"))
        n_train = int(result.get("n_train", 0))
        n_val = int(result.get("n_val", 0))
        n_test = int(result.get("n_test", 0))
        feat_dim = int(result.get("feature_dim", 0))

        BOX_W, BOX_H = 1.55, 1.55
        GAP = 0.38
        Y_CENTER = 2.0
        COLORS = ["#d0e8ff", "#d4f0d4", "#fff3cc", "#ffdad4", "#e8d4f0"]
        BORDER = ["#3a7fbf", "#3a9f3a", "#c8a000", "#bf4030", "#7040a0"]
        TITLE_COLOR = ["#1a4f8a", "#1a6a1a", "#7a5800", "#8a2018", "#4a1870"]

        total_w = 5 * BOX_W + 4 * GAP
        x_start = (10 - total_w) / 2

        nodes = [
            (
                "1. Raw HSI Data",
                [
                    f"Samples:  {n_train + n_val + n_test:,}",
                    f"Train:    {n_train:,}",
                    f"Val:      {n_val:,}",
                    f"Test:     {n_test:,}",
                    "Format:   ENVI (.hdr)",
                ],
            ),
            (
                "2. Preprocessing",
                [
                    f"Spectral: {spectral_norm.upper()}",
                    f"Scaling:  {global_scale.capitalize()}",
                    "Scope:    per-spectrum",
                    "Leak-safe: train-fit",
                ],
            ),
            (
                "3. Feature Eng.",
                [
                    f"Mode:     {feature_mode}",
                    f"PCA var:  {pca_comp}",
                    f"Feat dim: {feat_dim}",
                    "SVD-based PCA",
                ],
            ),
            (
                "4. Model Training",
                [
                    f"Models:   {n_models}",
                    "Library:  scikit-learn",
                    "Strategy: fit train",
                    "Select: val, report test",
                ],
            ),
            (
                "5. Evaluation",
                [
                    f"Best:  {best_model[:16]}",
                    "Metric: Macro F1",
                    "CM: normalised",
                    "Per-class P/R/F1",
                ],
            ),
        ]

        box_xs = [x_start + i * (BOX_W + GAP) for i in range(5)]

        for i, (title, lines) in enumerate(nodes):
            x0 = box_xs[i]
            y0 = Y_CENTER - BOX_H / 2

            box = FancyBboxPatch(
                (x0, y0), BOX_W, BOX_H,
                boxstyle="round,pad=0.06",
                facecolor=COLORS[i],
                edgecolor=BORDER[i],
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(box)

            ax.text(
                x0 + BOX_W / 2, y0 + BOX_H - 0.22,
                title,
                ha="center", va="top",
                fontsize=7.8, fontweight="bold",
                color=TITLE_COLOR[i], zorder=3,
            )

            ax.plot(
                [x0 + 0.12, x0 + BOX_W - 0.12],
                [y0 + BOX_H - 0.38, y0 + BOX_H - 0.38],
                color=BORDER[i], linewidth=0.8, zorder=3,
            )

            for j, line in enumerate(lines):
                ax.text(
                    x0 + 0.13, y0 + BOX_H - 0.52 - j * 0.255,
                    line,
                    ha="left", va="top",
                    fontsize=6.8, color="#222222",
                    family="monospace", zorder=3,
                )

            if i < 4:
                arrow_x0 = x0 + BOX_W
                arrow_x1 = box_xs[i + 1]
                mid_x = (arrow_x0 + arrow_x1) / 2
                arrow = FancyArrowPatch(
                    (arrow_x0, Y_CENTER), (arrow_x1, Y_CENTER),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color="#555555",
                    linewidth=1.3,
                    zorder=4,
                )
                ax.add_patch(arrow)
                ax.text(
                    mid_x, Y_CENTER + 0.15,
                    "→",
                    ha="center", va="bottom",
                    fontsize=8, color="#555555", zorder=5,
                )

        ax.text(
            5.0, 3.88,
            "HSI Classification — Methodology Pipeline",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="#1d1d1f",
        )
        ax.text(
            5.0, 0.08,
            "Figure: End-to-end hyperspectral image classification pipeline. "
            "All statistic-based transforms are computed from the training partition only.",
            ha="center", va="bottom",
            fontsize=7.5, color="#555555", style="italic",
        )

        fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.08)
        return fig

    def _draw_feature_importance_on_axis(
        self,
        ax: object,
        importances: np.ndarray,
        feature_names: list[str],
        model_name: str,
    ) -> None:
        """Horizontal bar chart of the top-25 most important features."""
        fi = np.asarray(importances, dtype=np.float64)
        n_show = min(25, len(fi))
        top_idx = np.argsort(fi)[::-1][:n_show][::-1]  # highest bar at top
        vals = fi[top_idx]
        names = [
            feature_names[i] if i < len(feature_names) else f"f{i}"
            for i in top_idx
        ]
        norm_vals = vals / (vals.max() + 1e-9)
        bar_colors = cm.YlOrRd(0.35 + 0.65 * norm_vals)
        ax.barh(range(n_show), vals, color=bar_colors,
                edgecolor="none", height=0.72)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(names, fontsize=6.2)
        ax.set_xlabel("Importance", fontsize=7.5)
        short_name = model_name.replace(
            "Classifier", "").replace("Regressor", "")
        ax.set_title(
            f"Feature Importance — {short_name}",
            fontsize=8,
            fontweight="bold",
        )
        ax.tick_params(axis="x", labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(left=0)

    def _build_source_split_table_figure(self, result: dict[str, object], cfg: Optional[PipelineInput]) -> Figure:
        fig = Figure(figsize=(11.69, 8.27), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.axis("off")

        dataset_text = cfg.dataset_path if cfg is not None else "N/A"
        rows = result.get("source_split_rows", [])
        ax.text(
            0.02,
            0.97,
            "Dataset Source Split Summary",
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.92,
            f"Dataset path: {dataset_text}",
            ha="left",
            va="top",
            fontsize=9,
        )

        if not rows:
            ax.text(0.5, 0.5, "No source split data available",
                    ha="center", va="center", fontsize=11)
            return fig

        total_train = int(sum(int(r[1]) for r in rows))
        total_val = int(sum(int(r[2]) for r in rows))
        total_test = int(sum(int(r[3]) for r in rows))
        total_all = int(sum(int(r[4]) for r in rows))

        table_rows: list[list[str]] = []
        for source_name, train_count, val_count, test_count, total_count, train_ratio, val_ratio, test_ratio in rows:
            table_rows.append(
                [
                    str(source_name),
                    f"{int(train_count):,}",
                    f"{int(val_count):,}",
                    f"{int(test_count):,}",
                    f"{int(total_count):,}",
                    f"{float(train_ratio) * 100.0:,.1f}%",
                    f"{float(val_ratio) * 100.0:,.1f}%",
                    f"{float(test_ratio) * 100.0:,.1f}%",
                ]
            )

        table_rows.append(
            [
                "TOTAL",
                f"{total_train:,}",
                f"{total_val:,}",
                f"{total_test:,}",
                f"{total_all:,}",
                "100.0%",
                "100.0%",
                "100.0%",
            ]
        )

        col_labels = [
            "Source (GT file)",
            "Train",
            "Validation",
            "Test",
            "Total",
            "Train Share",
            "Validation Share",
            "Test Share",
        ]

        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colLoc="center",
            bbox=[0.01, 0.08, 0.98, 0.78],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.22)

        n_rows = len(table_rows)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e8eef7")
                cell.set_text_props(weight="bold")
            if row == n_rows:
                cell.set_facecolor("#f2f2f2")
                cell.set_text_props(weight="bold")
            if col == 0 and row > 0:
                cell.set_text_props(ha="left")

        return fig

    def _apply_pdf_page_margins(
        self,
        fig: Figure,
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
        wspace: float = 0.26,
        hspace: float = 0.28,
    ) -> None:
        fig.set_constrained_layout(False)
        fig.subplots_adjust(left=left, right=right, top=top,
                            bottom=bottom, wspace=wspace, hspace=hspace)

    def _build_train_test_figure(self, result: dict[str, object]) -> Figure:
        split_stats = result.get("split_stats")
        split_preview = result.get("split_preview")
        raw_stats = result.get("raw_spectrum_stats")
        label_names = result.get("label_names") or {}
        fig = Figure(figsize=(12.0, 8.2), constrained_layout=False)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[
                              0.95, 1.05], hspace=0.32, wspace=0.28)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        if not split_stats:
            ax.axis("off")
            ax.text(0.5, 0.5, "Train/Test split stats are not available",
                    ha="center", va="center")
            ax2.axis("off")
            ax3.axis("off")
            ax4.axis("off")
            return fig

        labels = np.asarray(split_stats["labels"], dtype=np.int32)
        train_counts = np.asarray(split_stats["train_counts"], dtype=np.int32)
        test_counts = np.asarray(split_stats["test_counts"], dtype=np.int32)

        x = np.arange(labels.size)
        width = 0.42
        ax.bar(x - width / 2, train_counts, width=width,
               label="Train", color="#2a7fff")
        ax.bar(x + width / 2, test_counts, width=width,
               label="Test", color="#ff8a2a")

        ax.set_title("Train/Test Class Distribution")
        ax.set_xlabel("Class (pixel value)")
        ax.set_ylabel("Sample count")
        ax.set_xticks(x)
        ax.set_xticklabels([_format_class_label(int(v), label_names)
                           for v in labels], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.legend(loc="upper right")

        if not split_preview:
            ax2.axis("off")
            ax2.text(0.5, 0.5, "2D split preview is not available",
                     ha="center", va="center")
        else:
            train_xy = np.asarray(split_preview["train_xy"], dtype=np.float32)
            test_xy = np.asarray(split_preview["test_xy"], dtype=np.float32)
            train_labels = np.asarray(
                split_preview["train_labels"], dtype=np.int32)
            test_labels = np.asarray(
                split_preview["test_labels"], dtype=np.int32)
            labels_all = np.asarray(split_preview["labels"], dtype=np.int32)

            if train_xy.size == 0 and test_xy.size == 0:
                ax2.axis("off")
                ax2.text(0.5, 0.5, "2D split preview is empty",
                         ha="center", va="center")
            else:
                cmap = _build_class_cmap(labels_all)
                label_to_idx = {int(lbl): idx for idx,
                                lbl in enumerate(labels_all)}

                if train_xy.size > 0:
                    train_color_idx = np.array(
                        [label_to_idx[int(v)] for v in train_labels], dtype=np.int32)
                    ax2.scatter(
                        train_xy[:, 0],
                        train_xy[:, 1],
                        c=train_color_idx,
                        cmap=cmap,
                        s=10,
                        marker="o",
                        alpha=0.55,
                        edgecolors="none",
                        label="Train",
                    )

                if test_xy.size > 0:
                    test_color_idx = np.array(
                        [label_to_idx[int(v)] for v in test_labels], dtype=np.int32)
                    ax2.scatter(
                        test_xy[:, 0],
                        test_xy[:, 1],
                        c=test_color_idx,
                        cmap=cmap,
                        s=18,
                        marker="^",
                        alpha=0.8,
                        linewidths=0.2,
                        edgecolors="#222222",
                        label="Test",
                    )

                ax2.set_title("Train/Test 2D Split Preview")
                ax2.set_xlabel("Component 1")
                ax2.set_ylabel("Component 2")
                ax2.grid(linestyle="--", linewidth=0.6, alpha=0.28)
                ax2.legend(loc="best")

        self._draw_raw_spectrum_axis(
            ax3, raw_stats, split_name="train", label_names=label_names)
        self._draw_raw_spectrum_axis(
            ax4, raw_stats, split_name="test", label_names=label_names)

        return fig

    def _draw_raw_spectrum_axis(
        self,
        ax,
        raw_stats: Optional[dict[str, object]],
        split_name: str,
        label_names: dict[int, str],
    ) -> None:
        if not raw_stats:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"Raw spectrum ({split_name}) unavailable",
                ha="center",
                va="center",
            )
            return

        wavelengths = np.asarray(raw_stats.get(
            "wavelengths", []), dtype=np.float32)
        labels = np.asarray(raw_stats.get("labels", []), dtype=np.int32)
        class_means = np.asarray(raw_stats.get(
            f"{split_name}_class_mean", []), dtype=np.float32)
        class_support = np.asarray(raw_stats.get(
            f"{split_name}_class_support", []), dtype=np.int32)

        if class_means.size == 0 or labels.size == 0:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"Raw spectrum ({split_name}) unavailable",
                ha="center",
                va="center",
            )
            return

        n_bands = int(class_means.shape[1])
        if wavelengths.size == 0:
            wavelengths = np.arange(n_bands, dtype=np.float32)
        x = wavelengths[:n_bands]

        cmap = _build_class_cmap(labels)
        for i, cls in enumerate(labels):
            if i >= class_means.shape[0]:
                break
            if i < class_support.size and int(class_support[i]) <= 0:
                continue
            y = class_means[i]
            label_text = _format_class_label(int(cls), label_names)
            if i < class_support.size:
                label_text += f" (n={int(class_support[i])})"
            ax.plot(x, y, color=cmap(i), linewidth=1.35,
                    alpha=0.92, label=label_text)

        title = "Raw Spectrum by Class - Train" if split_name == "train" else "Raw Spectrum by Class - Test"
        ax.set_title(title)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Reflectance / Intensity")
        ax.grid(linestyle="--", linewidth=0.6, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    def _format_model_metrics_text(
        self,
        model_name: str,
        metrics: dict[str, object],
        label_names: dict[int, str],
    ) -> str:
        labels = np.asarray(metrics["labels"])
        precision = np.asarray(metrics["per_class_precision"])
        recall = np.asarray(metrics["per_class_recall"])
        f1 = np.asarray(metrics["per_class_f1"])
        support = np.asarray(metrics["support"])

        lines = [
            f"Model: {model_name}",
            "=" * 56,
            "Metrics set    : test split",
            f"Accuracy        : {metrics['accuracy']:.4f}",
            f"Macro Precision : {metrics['macro_precision']:.4f}",
            f"Macro Recall    : {metrics['macro_recall']:.4f}",
            f"Macro F1        : {metrics['macro_f1']:.4f}",
            "",
            "Per-class metrics",
            "Class | Precision | Recall | F1 | Support",
            "-" * 56,
        ]
        for i, cls in enumerate(labels):
            class_label = _format_class_label(int(cls), label_names)
            lines.append(
                f"{class_label:>14} | {precision[i]:>9.4f} | {recall[i]:>6.4f} | {f1[i]:>6.4f} | {int(support[i]):>7}"
            )
        return "\n".join(lines)


# ---------------------------
# Pipeline helper functions
# ---------------------------

def _list_available_model_entries(k_neighbors: int) -> list[tuple[str, str]]:
    return [(_categorize_model_name(name), name) for name, _ in _build_sklearn_model_specs([], k_neighbors)]


def _load_label_names(dataset_path: str) -> tuple[dict[int, str], np.ndarray]:
    root = Path(dataset_path)

    # Primary: COCO GT JSON categories (preferred)
    coco_label_names: dict[int, str] = {}
    coco_label_order: list[int] = []
    gt_dir = root / "GT"

    if gt_dir.exists() and gt_dir.is_dir():
        json_files = sorted(gt_dir.glob("*.json"))
        for json_path in json_files:
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            categories = data.get("categories")
            if not isinstance(categories, list):
                continue

            for category in categories:
                if not isinstance(category, dict):
                    continue
                cat_id = category.get("id")
                cat_name = category.get("name")
                if cat_id is None or cat_name is None:
                    continue

                try:
                    cat_id = int(cat_id)
                except Exception:
                    continue

                cat_name = str(cat_name).strip()
                if cat_id not in coco_label_names:
                    coco_label_names[cat_id] = cat_name
                    coco_label_order.append(cat_id)

        if coco_label_names:
            return coco_label_names, np.asarray(coco_label_order, dtype=np.int32)

    # New behavior: label names are strictly loaded from COCO GT JSON categories.
    # If no categories are found, return empty mapping.
    return {}, np.array([], dtype=np.int32)


def _parse_label_names_file(path: Path) -> tuple[dict[int, str], np.ndarray]:
    label_names: dict[int, str] = {}
    label_order: list[int] = []
    fallback_index = 0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8-sig",
                               errors="ignore").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        label_id, label_name = _parse_label_name_line(line, fallback_index)
        label_names[int(label_id)] = label_name
        label_order.append(int(label_id))
        fallback_index = max(fallback_index, int(label_id) + 1)

    return label_names, np.asarray(label_order, dtype=np.int32)


def _parse_label_name_line(line: str, fallback_index: int) -> tuple[int, str]:
    normalized = re.sub(r"[,\t|:]+", " ", line).strip()
    if not normalized:
        return fallback_index, str(fallback_index)

    parts = normalized.split(maxsplit=1)
    first = parts[0]
    if first.lstrip("+-").isdigit():
        label_id = int(first)
        label_name = parts[1].strip() if len(parts) > 1 else str(label_id)
        return label_id, label_name or str(label_id)

    return fallback_index, line


def _format_class_label(label: int, label_names: dict[int, str]) -> str:
    name = label_names.get(int(label))
    if not name:
        return str(int(label))

    name = str(name).strip()
    if not name or name == str(int(label)):
        return str(int(label))

    return f"{int(label)} ({name})"


def _format_class_name(label: int, label_names: dict[int, str]) -> str:
    name = label_names.get(int(label))
    if not name:
        return str(int(label))

    name = str(name).strip()
    return name if name else str(int(label))


def _build_metric_label_order(base_order: np.ndarray, *arrays: np.ndarray) -> np.ndarray:
    ordered: list[int] = []
    seen: set[int] = set()

    for lbl in np.asarray(base_order, dtype=np.int32).tolist():
        value = int(lbl)
        if value not in seen:
            ordered.append(value)
            seen.add(value)

    if arrays:
        data = np.concatenate([np.asarray(arr).reshape(-1) for arr in arrays])
        for lbl in np.unique(data).astype(np.int32, copy=False).tolist():
            value = int(lbl)
            if value not in seen:
                ordered.append(value)
                seen.add(value)

    return np.asarray(ordered, dtype=np.int32)


def _build_class_cmap(labels: np.ndarray) -> mcolors.ListedColormap:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    nonzero_labels = [int(lbl) for lbl in labels if int(lbl) != 0]
    if not nonzero_labels:
        return mcolors.ListedColormap([(0.0, 0.0, 0.0, 1.0)])

    jet = colormaps.get_cmap("jet")
    if len(nonzero_labels) == 1:
        nonzero_colors = [jet(0.5)]
    else:
        nonzero_colors = [jet(v) for v in np.linspace(
            0.05, 0.95, len(nonzero_labels))]

    color_list = []
    for lbl in labels:
        value = int(lbl)
        if value == 0:
            color_list.append((0.0, 0.0, 0.0, 1.0))
        else:
            color_list.append(nonzero_colors[nonzero_labels.index(value)])

    return mcolors.ListedColormap(color_list)


def _load_preview_category_names(gt_json_path: str) -> dict[int, str]:
    """Load COCO category ID → category name mapping from a GT JSON file."""
    if not os.path.exists(gt_json_path):
        return {}

    try:
        with open(gt_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    categories = data.get("categories")
    if not isinstance(categories, list):
        return {}

    category_map: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        cat_id = category.get("id")
        cat_name = category.get("name")
        if cat_id is None or cat_name is None:
            continue
        try:
            cat_id = int(cat_id)
        except Exception:
            continue
        cat_name = str(cat_name).strip()
        if cat_name:
            category_map[cat_id] = cat_name
    return category_map

def _build_sklearn_model_specs(
    selected_model_names: list[str],
    k_neighbors: int,
) -> list[tuple[str, ClassifierMixin]]:
    models: list[tuple[str, ClassifierMixin]] = []
    skip_names = {
        "ClassifierChain",
        "MultiOutputClassifier",
        "OneVsOneClassifier",
        "OneVsRestClassifier",
        "OutputCodeClassifier",
        "StackingClassifier",
        "VotingClassifier",
        "CategoricalNB",  # only for categorical features, not suitable for continuous spectra
        "ComplementNB",  # often fails with sparse input, and not ideal for pixel-wise classification
        # very slow and often fails to converge on high-dimensional data
        "GaussianProcessClassifier",
        "LabelPropagation",  # semi-supervised, not suitable for standard supervised classification
    }
    selected_set = set(selected_model_names)

    for name, cls in sorted(all_estimators(type_filter="classifier"), key=lambda x: x[0]):
        if name in skip_names:
            continue
        if selected_set and name not in selected_set:
            continue

        estimator = _safe_build_estimator(name, cls, k_neighbors=k_neighbors)
        if estimator is None:
            continue

        models.append((name, estimator))

    return models


def _categorize_model_name(model_name: str) -> str:
    if any(key in model_name for key in ["Logistic", "Linear", "Ridge", "SGD", "Perceptron", "PassiveAggressive"]):
        return "Linear"
    if any(key in model_name for key in ["SVC", "SVR", "NuSVC"]):
        return "Kernel"
    if any(key in model_name for key in ["Tree", "Forest", "Boost", "AdaBoost", "GradientBoost", "ExtraTrees"]):
        return "Tree Ensemble"
    if any(key in model_name for key in ["KNeighbors", "NearestCentroid", "RadiusNeighbors"]):
        return "Neighbors"
    if any(key in model_name for key in ["Gaussian", "Bernoulli", "Multinomial", "ComplementNB", "CategoricalNB"]):
        return "Naive Bayes"
    if any(key in model_name for key in ["MLP"]):
        return "Neural"
    if any(key in model_name for key in ["Discriminant"]):
        return "Discriminant"
    if any(key in model_name for key in ["Process"]):
        return "Gaussian Process"
    if any(key in model_name for key in ["Dummy"]):
        return "Baseline"
    return "Other"


def _safe_build_estimator(name: str, cls: type, k_neighbors: int) -> Optional[ClassifierMixin]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None

    required = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.default is inspect._empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            required.append(p.name)
    if required:
        logger.debug(
            "Skipping %s: required constructor params %s", name, required)
        return None

    kwargs = {}
    if "random_state" in sig.parameters:
        kwargs["random_state"] = 42
    if name == "KNeighborsClassifier" and "n_neighbors" in sig.parameters:
        kwargs["n_neighbors"] = max(1, int(k_neighbors))
    if name == "LogisticRegression" and "max_iter" in sig.parameters:
        kwargs["max_iter"] = 1000

    try:
        logger.debug("Instantiating estimator %s with kwargs=%s", name, kwargs)
        estimator = cls(**kwargs)
    except Exception as exc:
        logger.exception("Failed to instantiate %s: %s", name, exc)
        return None

    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        return None
    return estimator


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None) -> dict[str, object]:
    if labels is None or np.asarray(labels).size == 0:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = _build_metric_label_order(
            np.asarray(labels, dtype=np.int32), y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    prf = precision_recall_f1(y_true, y_pred, labels=labels)

    return {
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


def _build_feature_names(
    feature_mode: str,
    n_features: int,
    wavelengths: Optional[np.ndarray],
) -> list[str]:
    """Return human-readable feature names that match the feature vector layout."""
    _STAT = ["mean", "std", "min", "max", "ptp",
             "area", "l2", "mean_abs", "slope", "skew"]
    if feature_mode == "stat":
        return _STAT[:n_features]
    if feature_mode == "pca":
        return [f"PC{i + 1}" for i in range(n_features)]
    if feature_mode == "pca_stat":
        n_pca = max(0, n_features - len(_STAT))
        return _STAT + [f"PC{i + 1}" for i in range(n_pca)]
    # raw mode — label by wavelength if available
    if wavelengths is not None and len(wavelengths) == n_features:
        return [f"{float(w):.1f} nm" for w in wavelengths]
    return [f"Band {i}" for i in range(n_features)]


# ---------- Data processing visualization helpers (used by HSIPipelineWindow) ----------
def _sample_spectra_from_dataset(dataset_path: str, max_spectra: int):
    spectra, labels, source_files, wavelengths = HSI2D_loader(
        dataset_path=dataset_path, max_spectra=max_spectra
    )
    return spectra, labels, source_files, wavelengths


def _cubewise_dev_test_split(
    source_files: np.ndarray,
    test_ratio: float = 0.2,
    random_state: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by source cube so each cube belongs entirely to dev or test."""
    files = np.asarray(source_files)
    if files.ndim != 1:
        raise ValueError("source_files must be a 1D array")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1")

    unique_files = np.unique(files)
    if unique_files.size < 2:
        raise ValueError(
            "Dataset must contain at least 2 paired GT/HSI cubes "
            "(dev for train+val, and test). "
            f"Found {unique_files.size}."
        )

    rng = np.random.default_rng(random_state)
    shuffled = unique_files.copy()
    rng.shuffle(shuffled)

    n_test_files = int(np.floor(shuffled.size * test_ratio))
    if n_test_files <= 0:
        n_test_files = 1
    if n_test_files >= shuffled.size:
        n_test_files = shuffled.size - 1

    test_files = set(shuffled[:n_test_files].tolist())
    test_mask = np.array([f in test_files for f in files], dtype=bool)
    dev_mask = ~test_mask

    dev_idx = np.flatnonzero(dev_mask)
    test_idx = np.flatnonzero(test_mask)
    if dev_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Cube-wise split failed: dev/test indices are empty.")
    return dev_idx, test_idx


def _compute_split_stats(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.unique(np.concatenate([y_train, y_test]))
    train_counts = np.array([np.sum(y_train == label)
                            for label in labels], dtype=np.int64)
    test_counts = np.array([np.sum(y_test == label)
                           for label in labels], dtype=np.int64)
    return {
        "labels": labels.astype(np.int32, copy=False),
        "train_counts": train_counts,
        "test_counts": test_counts,
    }


def _compute_source_split_rows(
    source_files: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> list[tuple[str, int, int, int, int, float, float, float]]:
    names = np.asarray(source_files).astype(str)
    train_names = names[train_idx]
    val_names = names[val_idx]
    test_names = names[test_idx]

    all_sources = np.unique(np.concatenate(
        [train_names, val_names, test_names]))
    total_train = max(int(train_names.shape[0]), 1)
    total_val = max(int(val_names.shape[0]), 1)
    total_test = max(int(test_names.shape[0]), 1)

    rows: list[tuple[str, int, int, int, int, float, float, float]] = []
    for src in sorted(all_sources.tolist()):
        train_count = int(np.sum(train_names == src))
        val_count = int(np.sum(val_names == src))
        test_count = int(np.sum(test_names == src))
        total_count = train_count + val_count + test_count
        train_ratio = float(train_count / total_train)
        val_ratio = float(val_count / total_val)
        test_ratio = float(test_count / total_test)
        rows.append((src, train_count, val_count, test_count,
                    total_count, train_ratio, val_ratio, test_ratio))

    rows.sort(key=lambda x: x[4], reverse=True)
    return rows


def _compute_split_preview_payload(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_points_per_split: int = 4000,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)

    def _sample_xy(X: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        if X.shape[0] <= max_points:
            return X, y
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        return X[idx], y[idx]

    train_sample, y_train_sample = _sample_xy(
        X_train, y_train, max_points_per_split)
    test_sample, y_test_sample = _sample_xy(
        X_test, y_test, max_points_per_split)

    combined = np.vstack([train_sample, test_sample]
                         ).astype(np.float32, copy=False)
    if combined.shape[1] >= 2:
        centered = combined - np.mean(combined, axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            comp = vt[:2].T
            xy = centered @ comp
        except np.linalg.LinAlgError:
            xy = centered[:, :2]
    elif combined.shape[1] == 1:
        x1 = combined[:, 0]
        x2 = np.zeros_like(x1)
        xy = np.stack([x1, x2], axis=1)
    else:
        xy = np.zeros((combined.shape[0], 2), dtype=np.float32)

    n_train = train_sample.shape[0]
    train_xy = xy[:n_train]
    test_xy = xy[n_train:]
    labels = np.unique(np.concatenate([y_train_sample, y_test_sample])).astype(
        np.int32, copy=False)

    return {
        "train_xy": train_xy.astype(np.float32, copy=False),
        "test_xy": test_xy.astype(np.float32, copy=False),
        "train_labels": y_train_sample.astype(np.int32, copy=False),
        "test_labels": y_test_sample.astype(np.int32, copy=False),
        "labels": labels,
    }


def _compute_raw_spectrum_stats(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    wavelengths: Optional[np.ndarray],
    max_samples_per_split: int = 3000,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)

    def _sample_rows(X: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if X.shape[0] <= max_samples:
            return X, y
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        return X[idx], y[idx]

    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    train_sample, y_train_sample = _sample_rows(
        X_train_raw, y_train, max_samples_per_split)
    test_sample, y_test_sample = _sample_rows(
        X_test_raw, y_test, max_samples_per_split)
    train_sample = train_sample.astype(np.float32, copy=False)
    test_sample = test_sample.astype(np.float32, copy=False)

    labels = np.unique(np.concatenate([y_train, y_test]))

    n_bands = 0
    if train_sample.size:
        n_bands = int(train_sample.shape[1])
    elif test_sample.size:
        n_bands = int(test_sample.shape[1])

    if wavelengths is None:
        wl = np.arange(n_bands, dtype=np.float32)
    else:
        wl = np.asarray(wavelengths, dtype=np.float32)

    train_class_mean, train_class_support = _compute_classwise_spectrum_mean(
        train_sample, y_train_sample, labels)
    test_class_mean, test_class_support = _compute_classwise_spectrum_mean(
        test_sample, y_test_sample, labels)

    return {
        "wavelengths": wl,
        "labels": labels.astype(np.int32, copy=False),
        "train_class_mean": train_class_mean,
        "test_class_mean": test_class_mean,
        "train_class_support": train_class_support,
        "test_class_support": test_class_support,
    }


def _compute_classwise_spectrum_mean(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return np.zeros((labels.size, 0), dtype=np.float32), np.zeros(labels.size, dtype=np.int32)

    means = []
    supports = []
    for label in labels:
        mask = y == label
        supports.append(int(np.sum(mask)))
        if not np.any(mask):
            means.append(np.zeros(X.shape[1], dtype=np.float32))
            continue
        means.append(np.mean(X[mask], axis=0).astype(np.float32, copy=False))

    return np.asarray(means, dtype=np.float32), np.asarray(supports, dtype=np.int32)


def _select_representative_sample_name(source_files: np.ndarray, indices: np.ndarray) -> Optional[str]:
    names = _select_all_sample_names(source_files, indices)
    return names[0] if names else None


def _select_all_sample_names(source_files: np.ndarray, indices: np.ndarray) -> list[str]:
    """Return all unique sample names (in order of first appearance) for the given split indices."""
    files = np.asarray(source_files).astype(str)
    if indices.size == 0:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for file_name in files[np.asarray(indices, dtype=np.int64)]:
        base = os.path.splitext(os.path.basename(str(file_name)))[0]
        # strip annotation suffix (e.g. "curcumin001-FromSelection#3" -> "curcumin001-FromSelection")
        base = base.split("#")[0]
        base = os.path.splitext(base)[0]  # strip .json if present
        if base and base not in seen:
            result.append(base)
            seen.add(base)

    return result


def _build_split_annotation_mask(
    dataset_path: str,
    sample_name: str,
    split_source_ids: set[str],
    gt_shape: tuple[int, int],
) -> np.ndarray:
    """Return a boolean mask covering only the annotation pixels whose
    source ID (GT_filename#ann_id) appears in split_source_ids."""
    gt_dir = os.path.join(dataset_path, "GT")
    mask = np.zeros(gt_shape, dtype=bool)

    def _apply_json(json_path: str) -> None:
        source_name = os.path.basename(json_path)
        try:
            ann_list = _load_coco_json_masks(json_path)
        except Exception:
            return
        for ann_mask, image_name, _cat, ann_id in ann_list:
            base = os.path.splitext(image_name)[0]
            if base != sample_name:
                continue
            src_base = os.path.splitext(source_name)[0]
            sid_json = f"{source_name}#{ann_id}"
            sid_simple = f"{src_base}#{ann_id}"
            if sid_json in split_source_ids or sid_simple in split_source_ids:
                mask[:] |= ann_mask.astype(bool)

    # Prefer the JSON named after the sample
    candidate = os.path.join(gt_dir, sample_name + ".json")
    if os.path.exists(candidate):
        _apply_json(candidate)
    else:
        # Search all JSON files in GT/
        for entry in sorted(os.scandir(gt_dir), key=lambda e: e.name):
            if entry.name.lower().endswith(".json"):
                _apply_json(entry.path)

    return mask


def _build_split_reports(
    dataset_path: str,
    preprocessor: HSIPreprocessor,
    feature_extractor: HSIFeatureExtractor,
    source_files: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    X_train_feat: np.ndarray,
    X_val_feat: np.ndarray,
    X_test_feat: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    estimator,
    metric_labels: np.ndarray,
    best_model_name: str,
    label_names: dict[int, str],
) -> dict[str, list[dict[str, object]]]:
    split_specs = [
        ("Train", "train", train_idx, X_train_feat, y_train),
        ("Validation", "validation", val_idx, X_val_feat, y_val),
        ("Test", "test", test_idx, X_test_feat, y_test),
    ]

    reports: dict[str, list[dict[str, object]]] = {}
    fallback_samples = _list_dataset_pair_names(dataset_path)
    src_str = np.asarray(source_files, dtype=str)

    for page_label, split_key, split_idx, split_X, split_y in split_specs:
        split_pred = estimator.predict(split_X)
        split_metrics = _compute_metrics(split_y, split_pred, metric_labels)

        sample_names = _select_all_sample_names(source_files, split_idx)
        if not sample_names:
            sample_names = fallback_samples[:1] if fallback_samples else [
                "N/A"]

        report_list: list[dict[str, object]] = []
        for sname in sample_names:
            if sname != "N/A":
                loaded_name, sample_rgb, gt_map, sample_feat = _prepare_sample_preview_data(
                    dataset_path=dataset_path,
                    preprocessor=preprocessor,
                    feature_extractor=feature_extractor,
                    sample_name=sname,
                )
                pred_pixels = estimator.predict(sample_feat)
                pred_map = pred_pixels.reshape(
                    gt_map.shape).astype(np.int32, copy=False)
                # Predict on ALL annotated pixels of this sample (all splits combined)
                # so the map shows the full spatial coverage of the sample.
                pred_map[gt_map == 0] = 0

                # Compute per-sample metrics: filter split data to this sample's pixels
                split_src_ids = src_str[np.asarray(split_idx, dtype=np.int64)]
                sample_strip = np.array([s.split("#")[0]
                                        for s in split_src_ids])
                smask = sample_strip == sname
                if smask.any():
                    samp_metrics = _compute_metrics(
                        split_y[smask], split_pred[smask], metric_labels)
                else:
                    samp_metrics = split_metrics
            else:
                loaded_name = "N/A"
                sample_rgb = np.zeros((32, 32, 3), dtype=np.float32)
                gt_map = np.zeros((32, 32), dtype=np.int32)
                pred_map = np.zeros((32, 32), dtype=np.int32)
                samp_metrics = split_metrics

            report_list.append({
                "page_label": page_label,
                "sample_name": sname,
                "sample_rgb": sample_rgb,
                "gt_map": gt_map,
                "pred_map": pred_map,
                "metrics": samp_metrics,
                "best_model_name": best_model_name,
                "label_names": label_names,
            })

        reports[split_key] = report_list

    return reports


def _prepare_sample_preview_data(
    dataset_path: str,
    preprocessor: Optional[HSIPreprocessor] = None,
    feature_extractor: Optional[HSIFeatureExtractor] = None,
    sample_name: Optional[str] = None,
) -> tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    gt_dir = os.path.join(dataset_path, "GT")
    hsi_dir = os.path.join(dataset_path, "HSI")
    if not os.path.isdir(gt_dir) or not os.path.isdir(hsi_dir):
        raise ValueError("Dataset must contain GT and HSI folders")

    pair_name = None
    hdr_path = None

    gt_map = None
    if sample_name:
        candidate_gt_json = os.path.join(gt_dir, sample_name + ".json")
        candidate_hdr = _resolve_hsi_header_path(hsi_dir, sample_name)

        if os.path.exists(candidate_gt_json) and os.path.exists(candidate_hdr):
            pair_name = sample_name
            hdr_path = candidate_hdr
            masks = _load_coco_json_masks(candidate_gt_json)
            if masks:
                # combine all annotation masks for this selected sample image
                combined_map = None
                for mask_arr, image_name, category_id, ann_id in masks:
                    if os.path.splitext(image_name)[0] != sample_name:
                        continue
                    if combined_map is None:
                        combined_map = np.zeros_like(mask_arr, dtype=np.int32)
                    combined_map[mask_arr.astype(bool)] = int(category_id)
                if combined_map is not None:
                    gt_map = combined_map
                else:
                    # fallback to first sample in JSON
                    gt_map = masks[0][0].astype(np.int32, copy=False)
        else:
            # Fall back to auto selection if previously selected sample no longer exists.
            sample_name = None

    if not sample_name:
        # JSON only: try COCO GT files
        for entry in sorted(os.scandir(gt_dir), key=lambda e: e.name):
            if not entry.name.lower().endswith(".json"):
                continue
            masks = _load_coco_json_masks(entry.path)
            by_image = {}
            for mask_arr, image_name, category_id, ann_id in masks:
                base = os.path.splitext(image_name)[0]
                by_image.setdefault(base, []).append((mask_arr, category_id))

            for base, ann_list in by_image.items():
                candidate = _resolve_hsi_header_path(hsi_dir, base)
                if not os.path.exists(candidate):
                    continue
                pair_name = base
                hdr_path = candidate
                combined_map = np.zeros_like(ann_list[0][0], dtype=np.int32)
                for mask_arr, category_id in ann_list:
                    combined_map[mask_arr.astype(bool)] = int(category_id)
                gt_map = combined_map
                break
            if pair_name is not None:
                break

        # If no PNG found, try JSON COCO masks
        if pair_name is None:
            for entry in sorted(os.scandir(gt_dir), key=lambda e: e.name):
                if not entry.name.lower().endswith(".json"):
                    continue
                masks = _load_coco_json_masks(entry.path)
                for mask_arr, image_name, category_id, ann_id in masks:
                    base = os.path.splitext(image_name)[0]
                    candidate = _resolve_hsi_header_path(hsi_dir, base)
                    if os.path.exists(candidate):
                        pair_name = base
                        gt_map = mask_arr.astype(np.int32, copy=False)
                        hdr_path = candidate
                        break
                if pair_name is not None:
                    break

    if pair_name is None or hdr_path is None:
        raise ValueError("No GT/HSI pair found for preview")

    if gt_map is None:
        raise ValueError("No JSON GT map available for preview sample")

    cube = spy.open_image(hdr_path)
    cube_arr = np.asarray(cube.open_memmap(), dtype=np.float32)
    if cube_arr.ndim != 3:
        raise ValueError("HSI cube must have shape (H, W, C)")

    metadata_wavelengths = _parse_wavelength_metadata(
        getattr(cube, "metadata", None))

    h, w, c = cube_arr.shape
    if gt_map.shape[0] != h or gt_map.shape[1] != w:
        raise ValueError("GT and HSI size mismatch in preview sample")

    sample_rgb = _make_pseudo_rgb(cube_arr, metadata_wavelengths)
    pixels = cube_arr.reshape(-1, c)
    if preprocessor is not None and feature_extractor is not None:
        pixels_pre = preprocessor.transform(pixels)
        pixels_feat = feature_extractor.transform(pixels_pre)
    else:
        pixels_feat = None

    return pair_name, sample_rgb, gt_map, pixels_feat


def _collect_sample_annotations_from_json(dataset_path: str, sample_name: str):
    gt_dir = os.path.join(dataset_path, "GT")
    if not os.path.isdir(gt_dir):
        return []

    collected = []
    for entry in sorted(os.scandir(gt_dir), key=lambda e: e.name):
        if not entry.name.lower().endswith(".json"):
            continue
        try:
            masks = _load_coco_json_masks(entry.path)
        except Exception:
            continue
        for mask_arr, image_name, category_id, ann_id in masks:
            if os.path.splitext(image_name)[0] == sample_name:
                collected.append((mask_arr, category_id, ann_id))
    return collected


def _make_pseudo_rgb(cube: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
    _, _, c = cube.shape
    if c < 3:
        gray = cube[:, :, 0]
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        band_indices = _select_rgb_band_indices(c, wavelengths)
        r_idx, g_idx, b_idx = band_indices
        rgb = np.stack([cube[:, :, r_idx], cube[:, :, g_idx],
                       cube[:, :, b_idx]], axis=2)

    p2 = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    p98 = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    denom = np.maximum(p98 - p2, 1e-8)
    out = (rgb - p2) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _parse_wavelength_metadata(metadata: Optional[dict[str, object]]) -> Optional[np.ndarray]:
    if not metadata:
        return None

    raw = metadata.get("wavelength")
    if raw is None:
        return None

    try:
        wavelengths = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        cleaned: list[float] = []
        for value in raw if isinstance(raw, (list, tuple, np.ndarray)) else []:
            text = str(value).strip()
            if not text:
                continue
            try:
                cleaned.append(float(text))
            except ValueError:
                continue
        wavelengths = np.asarray(cleaned, dtype=np.float32)

    return wavelengths if wavelengths.size > 0 else None


def _select_rgb_band_indices(n_bands: int, wavelengths: Optional[np.ndarray] = None) -> tuple[int, int, int]:
    if n_bands <= 0:
        return 0, 0, 0
    if n_bands == 1:
        return 0, 0, 0
    if n_bands == 2:
        return 1, 0, 0

    if wavelengths is not None and np.asarray(wavelengths).size >= n_bands:
        wl = np.asarray(wavelengths, dtype=np.float32).reshape(-1)[:n_bands]
        target_order = np.array(RGB_TARGET_WAVELENGTHS, dtype=np.float32)
        remaining = list(range(n_bands))
        selected: list[int] = []

        for target in target_order:
            sorted_candidates = np.argsort(np.abs(wl - float(target))).tolist()
            chosen = None
            for candidate in sorted_candidates:
                if int(candidate) in remaining:
                    chosen = int(candidate)
                    break
            if chosen is None:
                chosen = int(sorted_candidates[0])
            selected.append(chosen)
            if chosen in remaining:
                remaining.remove(chosen)

        return int(selected[0]), int(selected[1]), int(selected[2])

    center_idx = n_bands // 2
    left_idx = max(0, center_idx - max(1, n_bands // 4))
    right_idx = min(n_bands - 1, center_idx + max(1, n_bands // 4))

    if left_idx == center_idx:
        left_idx = max(0, center_idx - 1)
    if right_idx == center_idx:
        right_idx = min(n_bands - 1, center_idx + 1)

    return int(right_idx), int(center_idx), int(left_idx)


def _list_dataset_pair_names(dataset_path: str) -> list[str]:
    if not dataset_path:
        return []

    gt_dir = os.path.join(dataset_path, "GT")
    hsi_dir = os.path.join(dataset_path, "HSI")
    if not os.path.isdir(gt_dir) or not os.path.isdir(hsi_dir):
        return []

    names: list[str] = []
    # JSON-based GT; each image is a sample candidate
    for entry in sorted(os.scandir(gt_dir), key=lambda e: e.name):
        if not entry.name.lower().endswith(".json"):
            continue
        try:
            with open(entry.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        for img in data.get('images', []):
            image_name = img.get('file_name')
            if not image_name:
                continue
            base = os.path.splitext(image_name)[0]
            if base in names:
                continue
            if os.path.exists(_resolve_hsi_header_path(hsi_dir, base)):
                names.append(base)

    return names


LIGHT_QSS = "light.qss"
DARK_QSS = "dark.qss"


def _qss_dir() -> Path:
    """
    หา directory ของ .qss ให้ถูกต้องทั้งตอน dev และตอน frozen (PyInstaller)
    ค้นหาตาม priority: pipeline/ → project root → exe folder
    """
    import sys
    if getattr(sys, "frozen", False):          # PyInstaller bundle
        return Path(sys.executable).parent
    # dev: ลองหาใน pipeline/ ก่อน ถ้าไม่มีให้ขึ้นไป project root
    script_dir = Path(__file__).parent
    if (script_dir / LIGHT_QSS).exists() or (script_dir / DARK_QSS).exists():
        return script_dir
    parent_dir = script_dir.parent
    if (parent_dir / LIGHT_QSS).exists() or (parent_dir / DARK_QSS).exists():
        return parent_dir
    return script_dir


def is_dark_mode() -> bool:
    palette = QApplication.palette()
    bg = palette.color(QPalette.ColorRole.Window)
    return bg.lightness() < 128


def apply_theme(app: QApplication, dark: bool) -> None:
    qss_path = _qss_dir() / (DARK_QSS if dark else LIGHT_QSS)
    if qss_path.exists():
        logger.info("Applying %s theme from %s",
                    "dark" if dark else "light", qss_path)
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))


class ThemeWatcher(QObject):
    """
    ฟัง ApplicationPaletteChange event แล้ว re-apply theme อัตโนมัติ
    เมื่อ user เปลี่ยน system light/dark mode ระหว่าง app รัน
    """

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app
        self._dark = is_dark_mode()
        app.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ApplicationPaletteChange:
            dark_now = is_dark_mode()
            if dark_now != self._dark:          # เปลี่ยนจริง ไม่ใช่แค่ palette refresh
                self._dark = dark_now
                apply_theme(self._app, dark_now)
        return False


def main() -> None:
    app = QApplication([])
    app.setStyle("Fusion")
    window = HSIPipelineWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
