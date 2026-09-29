# Agent instructions: thesis feature extractor

## Purpose

This repository is the separate thesis feature-extraction pipeline. It is not
the compact Elective 3 extractor and must not be replaced by it.

## Before running

Read `feature_contract.md` and review the current draft status in `README.md`.
Use `make_feature_debug.py` on a small sample before any large extraction.

## Data contract

The primary unit is one receipt image. Crop regions are intermediate audit
measurements and must not become independent classifier examples.

The extractor is nonsemantic: it does not use OCR text, spelling, language,
topic, or word meaning.

## Safety

Do not run full-dataset extraction or change feature definitions without an
explicit thesis decision. Keep generated debug files, datasets, and Python
environments out of Git. Preserve the independent Elective extractor.
