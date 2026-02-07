# Shezhenv3-COCO Tongue Diagnosis Framework

## Goal
Build a modular detection pipeline for tongue diagnosis using the shezhenv3-coco dataset.
Primary output: a clean, extensible detection stack with standardized interfaces.

## Repository layout
- `datasets/`: dataset loaders + COCO parser
- `models/`: backbone + detection heads + detector
- `augments/`: augmentation pipeline
- `engine/`: trainer + evaluator
- `configs/`: experiment configs (YAML/JSON)
- `tools/`: CLI entry points
- `tests/`: unit tests per module

## Core interfaces (minimal)
Dataset:
- `__len__()`
- `__getitem__(idx)` -> `(image, target)`
  - `target` uses COCO-style dict: `boxes`, `labels`, `image_id`, `area`, `iscrowd`

Backbone:
- `forward(images)` -> `features`

DetectionHead:
- `forward(features, targets=None)` -> `losses` (train) or `detections` (eval)

Detector:
- `forward(images, targets=None)` -> `losses` or `detections`

Trainer:
- `train_one_epoch(...)`
- `evaluate(...)`

Augment:
- `__call__(image, target)` -> `(image, target)`

## Data conventions
- `image`: torch float32, CHW, range [0,1]
- `boxes`: float32 tensor Nx4 (xmin, ymin, xmax, ymax)
- `labels`: int64 tensor N
- `image_id`: int64 scalar

## How to run (in cv conda env)
- Run tests: `conda run -n cv pytest -q`
- Training (later): `conda run -n cv python tools/train.py --config configs/base.yaml`
- Eval (later): `conda run -n cv python tools/eval.py --config configs/base.yaml`

## Notes
- Prefer minimal dependencies (torch/torchvision only).
- Keep tests small and fast (1-2 images) to support quick iteration.
- Record issues/tips in `codex.MD` as they appear.
