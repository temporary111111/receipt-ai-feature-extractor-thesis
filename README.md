# Receipt Feature Extractor

Standalone feature-extraction program for the machine-learning elective dataset.

This program is intentionally separate from the thesis receipt cropper. It reads
crop manifests and crop images, computes nonsemantic visual measurements, and
writes two CSV files:

- `image_features.csv`: one row per receipt, intended for the classifier dataset;
- `region_features.csv`: one row per crop, intended for audit and debugging.

The current implementation is a draft/pilot. It has not yet been approved for
full-dataset extraction. Review `feature_contract.md` before running it on all
receipts.

The primary unit of classification is one receipt image. Crops are intermediate
measurement regions and must not be split independently across train, validation,
and test partitions.

The extractor does not use OCR text, spelling, language, topic, or word meaning.

## Visual feature debugging

Use the companion audit script on a small receipt sample before a large run:

```powershell
python .\make_feature_debug.py `
  --crop-dir "...\receipt_0001" `
  --output-dir ".\debug\receipt_0001" `
  --max-crops 5
```

For each selected crop it writes the original crop, grayscale image, ink mask,
gradient map, edge map, local-variance map, and `debug_summary.csv`. These are
visual diagnostics; they are not additional classifier features.
