# Receipt Feature Contract

This document defines the first reproducible feature schema for the machine-learning elective dataset. It is deliberately nonsemantic: no OCR string, spelling, language, topic, or word meaning is used.

## Observation unit and labels

- The primary observation is one source receipt image.
- `label=1` means AI-generated under the dataset's source rules.
- `label=0` means non-AI-generated under the dataset's source rules.
- `class_name` stores the readable label (`ai_generated` or `non_ai_generated`).
- All crops from one source receipt stay in the same future train/validation/test partition.

## Outputs

`image_features.csv` is the classifier-ready table. It has one row per receipt and contains metadata, quality flags, and fixed-length aggregates of crop measurements.

`region_features.csv` is an audit table. It has one row per crop and retains crop geometry and per-crop measurements. It is not the primary classifier table.

## Image-level metadata and quality fields

- `image_id`, `source_image`, `source_path`
- `class_name`, `label`
- `split`: `train`, `validation`, or `test`; assigned at receipt level, never at crop level
- `source_width`, `source_height`
- `crop_count`, `has_text_regions`, `crop_area_ratio_sum`
- `quality_status`: `ok`, `low_crop_count`, `zero_crops`, `fallback_review`, `missing_crop_file`, or `source_missing`
- `include_default`: `1` only when `quality_status=ok`, otherwise `0`
- `dataset_version`, `cropper_version`, `feature_version`

Each crop manifest also records `detection_mode`: `normal`, `strict_fallback`, or `adaptive_local_contrast`. Any non-normal mode makes the image `fallback_review` until its localization is visually checked.

The 15-crop threshold is a review flag, not a claim that a receipt with fewer crops is invalid. Zero-crop images must be reviewed before they are used for model fitting. Metadata and version fields are for auditability and must not be used as classifier inputs.

## Per-crop feature groups

### Geometry

Normalized crop width, height, area, aspect ratio, center position, and recorded padding.

### Intensity and text occupancy

Grayscale mean, standard deviation, 10th/50th/90th percentiles, Otsu-based dark-ink fraction, ink bounding-box area ratio, and row/column occupancy statistics.

### Edges and sharpness

Gradient mean and standard deviation, edge density based on a fixed gradient threshold of 40 grayscale-intensity units, and Laplacian variance. The threshold is a draft parameter that must be frozen and recorded before final extraction.

### Texture

Mean 3x3 local variance and grayscale entropy.

## Aggregation

Every numeric per-crop feature is summarized at image level with:

`mean`, `std`, `min`, `max`, `median`, `q25`, and `q75`.

This produces a fixed-length representation despite each receipt having a different number of detected text regions. If no valid crop exists, aggregate values are blank and `include_default=0`; the row remains in the audit CSV but should be excluded from the initial classifier matrix until reviewed.

The primary image-level table must keep metadata separate from model inputs. The columns `image_id`, `source_image`, `source_path`, `class_name`, `label`, `split`, and version fields are identifiers, targets, or audit metadata. Only the defined numeric feature columns should be passed to a classifier.

The first feature-extraction implementation will support feature-group comparisons: geometry, intensity/occupancy, edges/sharpness, texture, and all-features. This allows the study to measure whether a feature family adds useful information rather than assuming that every candidate feature is helpful.

## Important limitations

The extractor measures visible pixel properties and crop geometry. It does not prove that a feature is an authenticity marker, and it does not use recognized text. Features affected by resolution, compression, acquisition method, font, layout, or crop errors must be checked as possible confounds before final modeling. Crop-area coverage is currently recorded as `crop_area_ratio_sum`; it is not an overlap-corrected union-area measurement.
