"""Extract nonsemantic visual features from receipt text crops.

The primary observation is one source receipt image.  Crop-level measurements
are aggregated into a fixed-length image-level record for later ML use.  A
second region-level CSV is written for auditability and debugging.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REGION_FEATURES = [
    "crop_width_norm",
    "crop_height_norm",
    "crop_aspect_ratio",
    "crop_area_norm",
    "crop_center_x_norm",
    "crop_center_y_norm",
    "crop_padding_px",
    "gray_mean",
    "gray_std",
    "gray_p10",
    "gray_median",
    "gray_p90",
    "ink_fraction",
    "ink_bbox_area_ratio",
    "row_occupancy_mean",
    "row_occupancy_std",
    "col_occupancy_mean",
    "col_occupancy_std",
    "gradient_mean",
    "gradient_std",
    "edge_density",
    "laplacian_variance",
    "local_variance_mean",
    "entropy",
]

# Fixed across crops so edge density remains comparable between samples.
# This is an initial draft parameter and should be frozen before final extraction.
EDGE_GRADIENT_THRESHOLD = 40.0


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    weighted = np.arange(256, dtype=np.float64) * hist
    cumulative_count = np.cumsum(hist)
    cumulative_weight = np.cumsum(weighted)
    denom = cumulative_count * (total - cumulative_count)
    numer = (cumulative_weight * total - cumulative_weight[-1] * cumulative_count) ** 2
    score = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    return int(np.argmax(score))


def safe_percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def region_pixel_features(image_path: Path) -> dict[str, float]:
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return {name: 0.0 for name in REGION_FEATURES[7:]}

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    threshold = otsu_threshold(gray_u8)
    # Otsu may classify a near-uniform crop as foreground.  Keep the mask
    # conservative for such cases so background is not treated as text ink.
    mask = gray_u8 <= threshold if gray_u8.std() >= 3.0 else np.zeros_like(gray_u8, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs):
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        bbox_ratio = bbox_area / float(gray.size)
    else:
        bbox_ratio = 0.0

    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.hypot(gx, gy)
    edge_density = float(np.mean(gradient > EDGE_GRADIENT_THRESHOLD)) if gradient.size else 0.0

    if gray.shape[0] > 2 and gray.shape[1] > 2:
        center = gray[1:-1, 1:-1]
        lap = (
            gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
            - 4.0 * center
        )
        lap_var = float(np.var(lap))
    else:
        lap_var = 0.0

    padded = np.pad(gray, 1, mode="edge")
    neighborhoods = np.stack(
        [padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
         for dy in range(3) for dx in range(3)], axis=0
    )
    local_var = np.var(neighborhoods, axis=0)
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / max(float(gray.size), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    row_occupancy = mask.mean(axis=1) if mask.size else np.array([0.0])
    col_occupancy = mask.mean(axis=0) if mask.size else np.array([0.0])
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_p10": safe_percentile(gray, 10),
        "gray_median": safe_percentile(gray, 50),
        "gray_p90": safe_percentile(gray, 90),
        "ink_fraction": float(mask.mean()),
        "ink_bbox_area_ratio": bbox_ratio,
        "row_occupancy_mean": float(row_occupancy.mean()),
        "row_occupancy_std": float(row_occupancy.std()),
        "col_occupancy_mean": float(col_occupancy.mean()),
        "col_occupancy_std": float(col_occupancy.std()),
        "gradient_mean": float(gradient.mean()),
        "gradient_std": float(gradient.std()),
        "edge_density": edge_density,
        "laplacian_variance": lap_var,
        "local_variance_mean": float(local_var.mean()),
        "entropy": entropy,
    }


def aggregate(values: Iterable[float], prefix: str) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {f"{prefix}_{stat}": None for stat in ("mean", "std", "min", "max", "median", "q25", "q75")}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std()),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q25": float(np.percentile(array, 25)),
        f"{prefix}_q75": float(np.percentile(array, 75)),
    }


def parse_root_specs(specs: list[str]) -> list[tuple[int, str, Path]]:
    parsed = []
    for spec in specs:
        label_text, path_text = spec.split("=", 1)
        label = int(label_text)
        class_name = "non_ai_generated" if label == 0 else "ai_generated" if label == 1 else f"class_{label}"
        parsed.append((label, class_name, Path(path_text)))
    return parsed


def process_receipt(
    receipt_dir: Path,
    source_dir: Path,
    label: int,
    class_name: str,
    split: str,
    dataset_version: str,
    cropper_version: str,
    feature_version: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    receipt_id = receipt_dir.name
    manifest_path = receipt_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline=""))) if manifest_path.exists() else []
    source_name = f"{receipt_id}.png"
    source_path = source_dir / source_name
    if not source_path.exists():
        jpg = source_dir / f"{receipt_id}.jpg"
        source_path = jpg if jpg.exists() else source_path
    source_missing = not source_path.exists()
    if source_missing:
        source_width, source_height = 0, 0
    else:
        with Image.open(source_path) as source_image:
            source_width, source_height = source_image.size
    source_area = max(source_width * source_height, 1)

    region_records: list[dict[str, object]] = []
    missing_crop_count = 0
    for row in rows:
        crop_path = receipt_dir / row["file"]
        if not crop_path.exists():
            missing_crop_count += 1
            continue
        x, y = int(row["source_x"]), int(row["source_y"])
        width, height = int(row["width"]), int(row["height"])
        record: dict[str, object] = {
            "image_id": receipt_id,
            "class_name": class_name,
            "label": label,
            "source_image": source_name,
            "crop_id": row["crop_id"],
            "source_x": x,
            "source_y": y,
            "width": width,
            "height": height,
            "crop_file": str(crop_path),
            "detection_mode": row.get("detection_mode", "normal"),
        }
        record.update({
            "crop_width_norm": width / max(source_width, 1),
            "crop_height_norm": height / max(source_height, 1),
            "crop_aspect_ratio": width / max(height, 1),
            "crop_area_norm": (width * height) / source_area,
            "crop_center_x_norm": (x + width / 2) / max(source_width, 1),
            "crop_center_y_norm": (y + height / 2) / max(source_height, 1),
            "crop_padding_px": float(row.get("padding_px", 0)),
        })
        record.update(region_pixel_features(crop_path))
        region_records.append(record)

    detection_modes = {str(r.get("detection_mode", "normal")) for r in region_records}
    if source_missing:
        quality_status = "source_missing"
    elif missing_crop_count:
        quality_status = "missing_crop_file"
    elif any(mode != "normal" for mode in detection_modes):
        quality_status = "fallback_review"
    elif not region_records:
        quality_status = "zero_crops"
    elif len(region_records) < 15:
        quality_status = "low_crop_count"
    else:
        quality_status = "ok"

    image_record: dict[str, object] = {
        "image_id": receipt_id,
        "source_image": source_name,
        "source_path": str(source_path),
        "class_name": class_name,
        "label": label,
        "split": split,
        "source_width": source_width,
        "source_height": source_height,
        "crop_count": len(region_records),
        "has_text_regions": int(bool(region_records)),
        "crop_area_ratio_sum": sum(float(r["crop_area_norm"]) for r in region_records),
        "quality_status": quality_status,
        # Only fully acceptable receipts enter the default classifier matrix.
        # Low/zero/missing-crop cases remain in the CSV for review.
        "include_default": int(quality_status == "ok"),
        "dataset_version": dataset_version,
        "cropper_version": cropper_version,
        "feature_version": feature_version,
    }
    for feature in REGION_FEATURES:
        image_record.update(aggregate((float(r[feature]) for r in region_records), feature))
    return image_record, region_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", action="append", required=True, help="LABEL=folder containing receipt_* crop folders; repeatable")
    parser.add_argument("--source-dir", action="append", required=True, help="LABEL=folder containing source receipt images; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="Optional development limit; 0 processes all receipts")
    parser.add_argument("--split", default="unassigned", choices=["train", "validation", "test", "unassigned"])
    parser.add_argument("--dataset-version", default="receipt-dataset-v1")
    parser.add_argument("--cropper-version", default="cropper-v1")
    parser.add_argument("--feature-version", default="features-v0.1-draft")
    args = parser.parse_args()
    source_map = {int(label): Path(path) for label, path in (s.split("=", 1) for s in args.source_dir)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_records: list[dict[str, object]] = []
    region_records: list[dict[str, object]] = []
    processed_images = 0
    for label, class_name, crop_root in parse_root_specs(args.crop_root):
        source_dir = source_map[label]
        for receipt_dir in sorted(crop_root.glob("receipt_*")):
            if receipt_dir.is_dir():
                image_record, records = process_receipt(
                    receipt_dir,
                    source_dir,
                    label,
                    class_name,
                    args.split,
                    args.dataset_version,
                    args.cropper_version,
                    args.feature_version,
                )
                image_records.append(image_record)
                region_records.extend(records)
                processed_images += 1
                if args.max_images and processed_images >= args.max_images:
                    break
        if args.max_images and processed_images >= args.max_images:
            break

    image_fields = list(image_records[0].keys()) if image_records else []
    region_fields = ["image_id", "class_name", "label", "source_image", "crop_id", "source_x", "source_y", "width", "height", "crop_file", "detection_mode"] + REGION_FEATURES
    with (args.output_dir / "image_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(image_records)
    with (args.output_dir / "region_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=region_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(region_records)
    print(f"images={len(image_records)} regions={len(region_records)}")


if __name__ == "__main__":
    main()
