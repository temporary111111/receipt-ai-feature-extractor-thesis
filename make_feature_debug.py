"""Create visual and numeric diagnostics for receipt crop features.

This is an audit tool, not a classifier. It makes the intermediate pixel
representations visible so a researcher can check whether the numerical
features correspond to text rather than background, shadows, or symbols.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from extract_features import EDGE_GRADIENT_THRESHOLD, otsu_threshold, region_pixel_features


def maps_for_crop(crop_path: Path) -> dict[str, Image.Image]:
    with Image.open(crop_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    threshold = otsu_threshold(gray_u8)
    mask = gray_u8 <= threshold if gray_u8.std() >= 3.0 else np.zeros_like(gray_u8, dtype=bool)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.hypot(gx, gy)
    edges = gradient > EDGE_GRADIENT_THRESHOLD
    padded = np.pad(gray, 1, mode="edge")
    neighborhoods = np.stack(
        [padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
         for dy in range(3) for dx in range(3)], axis=0
    )
    local_variance = np.var(neighborhoods, axis=0)

    def normalize(values: np.ndarray) -> Image.Image:
        low, high = float(values.min()), float(values.max())
        scaled = np.zeros_like(values, dtype=np.uint8) if high <= low else ((values - low) / (high - low) * 255).astype(np.uint8)
        return Image.fromarray(scaled, mode="L")

    return {
        "original": Image.open(crop_path).convert("RGB"),
        "grayscale": Image.fromarray(gray_u8, mode="L"),
        "ink_mask": Image.fromarray((mask.astype(np.uint8) * 255), mode="L"),
        "gradient_map": normalize(gradient),
        "edge_map": Image.fromarray((edges.astype(np.uint8) * 255), mode="L"),
        "local_variance_map": normalize(local_variance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-dir", type=Path, required=True, help="Receipt crop directory containing manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-crops", type=int, default=5)
    args = parser.parse_args()
    manifest_path = args.crop_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
    if args.max_crops > 0:
        rows = rows[:args.max_crops]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for row in rows:
        crop_path = args.crop_dir / row["file"]
        if not crop_path.exists():
            continue
        crop_output = args.output_dir / row["crop_id"]
        crop_output.mkdir(parents=True, exist_ok=True)
        for name, image in maps_for_crop(crop_path).items():
            image.save(crop_output / f"{name}.png")
        features = region_pixel_features(crop_path)
        summary_rows.append({"crop_id": row["crop_id"], "crop_file": str(crop_path), **features})

    if summary_rows:
        fields = list(summary_rows[0].keys())
        with (args.output_dir / "debug_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"debug_crops={len(summary_rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
