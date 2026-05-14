# -*- coding: utf-8 -*-
"""
Test: โหลดข้อมูลจาก dataset ด้วย HSI2D_loader
รันใน Spyder เพื่อดู variable ผ่าน Variable Explorer
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))

import numpy as np
import matplotlib.pyplot as plt
from preprocessing import (
    HSIPreprocessor,
    PreprocessConfig,
    stratified_train_val_split,
    encode_str_labels,
    samplewise_dev_test_split,
)
from hsi_loader import HSI2D_loader, _load_coco_json_masks, _resolve_hsi_header_path


DATASET_DIR = str(Path(__file__).parents[1] / "dataset")
GT_DIR = os.path.join(DATASET_DIR, "GT")
HSI_DIR = os.path.join(DATASET_DIR, "HSI")

# ── 1. ทดสอบหาไฟล์ HDR ──────────────────────────────────────────────────────
sample_name = "curcumin001"
hdr_path = _resolve_hsi_header_path(HSI_DIR, sample_name)
print(f"HDR path: {hdr_path}")
print(f"HDR exists: {os.path.exists(hdr_path)}")

# ── 2. โหลด COCO masks ──────────────────────────────────────────────────────
json_path = os.path.join(GT_DIR, "curcumin001.json")
masks = _load_coco_json_masks(json_path)
# masks = list of (mask_array, image_name, category_id, ann_id)
print(f"\nจำนวน annotations: {len(masks)}")
for mask_arr, image_name, category_id, ann_id in masks:
    print(f"  ann_id={ann_id}  class={category_id}  pixels={int(np.sum(mask_arr))}")

# ── 3. โหลด dataset ──────────────────────────────────────────────────────────
spectra, labels, filenames, wavelengths = HSI2D_loader(
    dataset_path=DATASET_DIR,
    max_spectra=500,
)
print(f"\nspectra.shape   : {spectra.shape}")
print(f"labels.shape    : {labels.shape}")
print(f"filenames.shape : {filenames.shape}")
print(f"wavelengths     : {len(wavelengths)} bands  [{wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm]")
print(f"unique classes  : {np.unique(labels).tolist()}")
print(f"unique files    : {np.unique(filenames).tolist()}")


print('==DEBUG==')

for cls in np.unique(labels):
    mask = labels == cls
    print(f"Class {cls}: {np.unique(filenames[mask])}")

# ── 4. แบ่งข้อมูล train/validation/test ตาม GUI default ────────────────────
labels_int, label_names, label_order = encode_str_labels(labels)

dev_idx, test_idx = samplewise_dev_test_split(
    labels=labels_int,
    source_files=filenames,
    test_ratio=0.2,
    random_state=42,
)
dev_labels = labels_int[dev_idx]
train_local_idx, val_local_idx = stratified_train_val_split(
    dev_labels,
    val_ratio=0.2,
    random_state=42,
)
train_idx = dev_idx[train_local_idx]
val_idx = dev_idx[val_local_idx]

print(f"\ntrain count      : {len(train_idx)}")
print(f"validation count : {len(val_idx)}")
print(f"test count       : {len(test_idx)}")
print(f"train classes    : {np.unique(labels[train_idx]).tolist()}")
print(f"val classes      : {np.unique(labels[val_idx]).tolist()}")
print(f"test classes     : {np.unique(labels[test_idx]).tolist()}")
print(f"train sources    : {np.unique(filenames[train_idx]).tolist()}")
print(f"val sources      : {np.unique(filenames[val_idx]).tolist()}")
print(f"test sources     : {np.unique(filenames[test_idx]).tolist()}")

# ── 5. Preprocess train/validation/test ตาม default GUI ───────────────────
pre_cfg = PreprocessConfig(
    remove_wavelength_ranges=[],
    spectral_normalization="snv",
    global_scaling="standard",
    clip_percentile=None,
)
pre = HSIPreprocessor(pre_cfg)

X_train_raw = spectra[train_idx]
X_val_raw = spectra[val_idx]
X_test_raw = spectra[test_idx]

X_train_pre = pre.fit_transform(X_train_raw, wavelengths=wavelengths)
X_val_pre = pre.transform(X_val_raw)
X_test_pre = pre.transform(X_test_raw)

print(f"\nX_train_pre.shape : {X_train_pre.shape}")
print(f"X_val_pre.shape   : {X_val_pre.shape}")
print(f"X_test_pre.shape  : {X_test_pre.shape}")

# ── 6. Plot average ± min/max spectra by label ─────────────────────────
unique_labels = np.unique(labels)
fig, ax = plt.subplots(figsize=(10, 5))
wl = np.asarray(wavelengths, dtype=float)
for i, label in enumerate(unique_labels):
    mask = labels == label
    if not np.any(mask):
        continue
    group = np.asarray(spectra[mask], dtype=float)
    mean_spec = np.mean(group, axis=0)
    min_spec = np.min(group, axis=0)
    max_spec = np.max(group, axis=0)
    color = plt.get_cmap("tab10")(i % 10)
    ax.plot(wl, mean_spec, color=color, linewidth=1.8, label=f"{label} mean")
    ax.fill_between(wl, min_spec, max_spec, color=color, alpha=0.18)

ax.set_title("Mean ± min/max spectra by label")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.4)
plt.tight_layout()
plt.show()
