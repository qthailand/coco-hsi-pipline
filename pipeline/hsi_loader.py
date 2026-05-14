from __future__ import annotations

import json
from typing import List, Tuple, Optional
import numpy as np
import os
import glob
import cv2
import logging
import spectral as spy

import matplotlib.pyplot as plt


def _decode_coco_segmentation(seg, width: int, height: int) -> np.ndarray:
    """Decode a COCO segmentation object into a (height, width) uint8 mask."""
    ann_mask = np.zeros((height, width), dtype=np.uint8)

    if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
        try:
            from pycocotools import mask as maskUtils
        except ImportError as e:
            raise ImportError(
                "pycocotools is required to decode RLE segmentation in COCO JSON") from e
        decoded = maskUtils.decode(seg)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        ann_mask = decoded.astype(np.uint8, copy=False)

    elif isinstance(seg, list):
        try:
            from pycocotools import mask as maskUtils
            rle = maskUtils.frPyObjects(seg, height, width)
            decoded = maskUtils.decode(rle)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            ann_mask = decoded.astype(np.uint8, copy=False)
        except Exception:
            # fallback to OpenCV polygon rasterization
            for poly in seg:
                poly_pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                if poly_pts.size == 0:
                    continue
                poly_int = np.round(poly_pts).astype(np.int32)
                cv2.fillPoly(ann_mask, [poly_int], 1)

    else:
        raise ValueError("Unsupported COCO segmentation format")

    return ann_mask


def _load_coco_json_masks(gt_json_path: str) -> List[tuple]:
    """Load COCO format GT JSON and generate per-annotation masks.

    Returns list of tuples: (mask_array, image_file_name, category_id, annotation_id)
    """
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])
    if not images:
        raise ValueError(f"COCO GT JSON has no images: {gt_json_path}")

    masks = []
    for ann in annotations:
        image_id = ann.get('image_id')
        if image_id is None or image_id not in images:
            continue

        img = images[image_id]
        width = img.get('width')
        height = img.get('height')
        if width is None or height is None:
            raise ValueError(
                f"Image width/height missing in COCO GT JSON: {gt_json_path}")

        seg = ann.get('segmentation')
        if seg is None:
            continue

        ann_mask = _decode_coco_segmentation(seg, width, height)

        if not np.any(ann_mask):
            _cat = ann.get('category_id', 0)
            raise ValueError(
                f"Empty mask from annotation {ann.get('id', -1)} "
                f"(image {img.get('file_name', '')}, class {_cat}). "
                "Mask decode aborted; verify GT segmentation coordinates and file size match HSI."
            )

        masks.append((ann_mask, img.get('file_name', ''), int(
            ann.get('category_id', 0)), int(ann.get('id', -1))))

    return masks


def load_coco_mask(gt_json_path: str) -> List[tuple]:
    """Public helper for loading COCO JSON masks from a path."""
    if not os.path.exists(gt_json_path):
        raise FileNotFoundError(f"COCO JSON file not found: {gt_json_path}")
    return _load_coco_json_masks(gt_json_path)


def _collect_global_categories(gt_path: str) -> dict:
    """Scan all COCO JSON files in gt_path and build a global name→normalized_id mapping.

    Category names are sorted alphabetically and assigned consecutive IDs starting at 1.
    This guarantees consistent integer labels even when different JSON files use
    conflicting category ID values for the same category name.

    Returns
    -------
    dict[str, int]
        {category_name: normalized_id}
    """
    names: set = set()
    for entry in os.scandir(gt_path):
        if not entry.name.endswith('.json'):
            continue
        try:
            with open(entry.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        for cat in data.get('categories', []):
            name = str(cat.get('name', '')).strip()
            if name:
                names.add(name)
    return {name: idx + 1 for idx, name in enumerate(sorted(names))}


def load_label_names(dataset_path: str) -> dict:
    """Return {normalized_id: category_name} for all categories across all GT JSON files.

    IDs are assigned by sorting category names alphabetically, so the mapping is
    consistent regardless of which ID each individual JSON file uses.

    Returns
    -------
    dict[int, str]
        {normalized_id: category_name}
    """
    gt_path = os.path.join(dataset_path, 'GT')
    if not os.path.isdir(gt_path):
        return {}
    name_to_id = _collect_global_categories(gt_path)
    return {v: k for k, v in name_to_id.items()}


def HSI2D_loader(
    dataset_path: str,
    max_spectra: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a 2D hyperspectral dataset and return labels, filenames, and wavelengths.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory or file.
    max_spectra : int, default=1000
        Maximum number of spectra (pixels) to sample. If the dataset has fewer,
        all available spectra are returned.
    Returns
    -------
    spectrals : np.ndarray, shape (N, C)
        Spectral features (N pixel, C bands).
    labels : np.ndarray, shape (N,), dtype=str
        Category name of each pixel.
    filenames : np.ndarray, shape (filename,), dtype=str
        filename of pixel.
    wavelengths : np.ndarray, shape (wavelengths,), dtype=float
        wavelengths of whole data
       """
    gt_path = os.path.join(dataset_path, 'GT')  # ground truth path
    hc_path = os.path.join(dataset_path, 'HSI')  # hyper cube path

    if not os.path.exists(gt_path):
        raise ValueError(
            f"{gt_path} doesn't exists, please check your dataset structure")
    if not os.path.exists(hc_path):
        raise ValueError(
            f"{hc_path} doesn't exists, please check your dataset structure")

    # load ground truth path first for pairing with hypercube

    spectrals = []
    filenames = []
    labels = []
    class_sample_sources: dict[str, set[str]] = {}

    # Iterate each annotation as a sample unit; this allows class-split on per-annotation sample.
    sample_annotations: dict[str,
                             list[tuple[np.ndarray, str, str, Optional[int]]]] = {}

    for gt_file in os.scandir(gt_path):
        if not gt_file.name.endswith('.json'):
            continue

        # Read this file's own category id→name map for remapping
        file_id_to_name: dict[int, str] = {}
        try:
            with open(gt_file.path, 'r', encoding='utf-8') as _f:
                _data = json.load(_f)
            for _cat in _data.get('categories', []):
                _cid = _cat.get('id')
                _cname = str(_cat.get('name', '')).strip()
                if _cid is not None and _cname:
                    file_id_to_name[int(_cid)] = _cname
        except Exception:
            pass

        for mask, image_name, category_id, ann_id in _load_coco_json_masks(gt_file.path):
            cat_name = file_id_to_name.get(int(category_id), str(category_id))
            sample_name = os.path.splitext(image_name)[0]
            sample_annotations.setdefault(sample_name, []).append(
                (mask.astype(np.int32, copy=False), cat_name, gt_file.name, ann_id)
            )

    for sample_name, annotations in sample_annotations.items():
        hc_file = _resolve_hsi_header_path(hc_path, sample_name)
        if not os.path.exists(hc_file):
            logging.warning(
                f'Skipping {sample_name}: HSI file not found ({hc_file})')
            continue

        logging.info(f'Processing sample: {sample_name} -> {hc_file}')
        hypercube = spy.open_image(hc_file)
        wavelengths = hypercube.metadata['wavelength']
        hc_mmap = hypercube.open_memmap()

        # ✅ Merge masks ต่อ label ก่อน
        h, w = hc_mmap.shape[:2]
        merged: dict[str, np.ndarray] = {}
        for ground_truth_array, annotation_cat, source_name, annotation_id in annotations:
            if annotation_cat not in merged:
                merged[annotation_cat] = np.zeros((h, w), dtype=np.uint8)
            merged[annotation_cat] |= (
                ground_truth_array == 1).astype(np.uint8)

        # ✅ แล้วค่อย sample ต่อ label
        for class_id, combined_mask in merged.items():
            pixel_coords = np.argwhere(combined_mask)
            logging.info(f'Found {len(pixel_coords)} pixels for label {
                         class_id} in {sample_name}')

            if len(pixel_coords) == 0:
                raise ValueError(
                    f"Label {class_id} in {
                        sample_name} has no pixels after merging masks."
                )

            if pixel_coords.shape[0] > max_spectra:
                rng = np.random.default_rng(
                    abs(hash(sample_name + class_id)) % (2**31))
                pixel_coords = pixel_coords[rng.choice(
                    pixel_coords.shape[0], max_spectra, replace=False)]

            spectral_vectors = np.array(
                hc_mmap[pixel_coords[:, 0], pixel_coords[:, 1], :])
            if spectral_vectors.ndim == 4:
                spectral_vectors = spectral_vectors.reshape(
                    len(pixel_coords), -1)

            logging.info(f"Reshaped spectral vectors to {
                         spectral_vectors.shape} for {sample_name}")

            spectrals.extend(spectral_vectors)
            class_sample_sources.setdefault(
                str(class_id), set()).add(sample_name)
            filenames.extend([sample_name] * len(pixel_coords))
            labels.extend([class_id] * len(pixel_coords))

    # Validate AFTER all GT files have been processed (not per-file).
    if len(spectrals) > 0:
        insufficient = []
        for class_id, sample_ids in class_sample_sources.items():
            if len(sample_ids) < 2:
                insufficient.append((str(class_id), len(sample_ids)))

        if insufficient:
            all_samples = set().union(*class_sample_sources.values()
                                      ) if class_sample_sources else set()
            if len(all_samples) < 2:
                logging.warning(
                    "Dataset has fewer than 2 distinct samples in total; "
                    "skipping strict per-class sample coverage check."
                )
            else:
                err_parts = [f"class {c} has {
                    n} sample(s) with pixels" for c, n in insufficient]
                raise ValueError(
                    "Each class must have at least 2 distinct pixel-containing samples. "
                    + "; ".join(err_parts)
                )

    return np.stack(spectrals), np.array(labels), np.array(filenames), np.array(wavelengths, dtype=float)


def _resolve_hsi_header_path(hsi_dir: str, sample_name: str) -> str:
    """Resolve ENVI header path for a sample.

    Supports both exact "sample.hdr" and variant names such as "sample.bsq.hdr".
    """
    # Try exact match first (e.g. sample.hdr)
    exact = os.path.join(hsi_dir, f"{sample_name}.hdr")
    if os.path.exists(exact):
        return exact

    # Try extension-only variants (e.g. sample.bsq.hdr, sample_10000_70.bsq.hdr).
    # The character immediately after sample_name must be '.' or '_' to avoid
    # accidentally matching a different sample that shares a common prefix
    # (e.g. abc001-s.bsq.hdr must NOT be returned when looking for abc001).
    pattern = os.path.join(hsi_dir, f"{sample_name}*.hdr")
    n = len(sample_name)
    candidates = sorted(
        c for c in glob.glob(pattern)
        if os.path.basename(c)[n:n + 1] in (".", "_")
    )
    if candidates:
        return candidates[0]

    # Return exact path as fallback so error messages are meaningful
    return exact
