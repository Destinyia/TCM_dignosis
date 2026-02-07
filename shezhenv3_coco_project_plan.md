# Shezhenv3-COCO Tongue Diagnosis Model - Development Plan

## 0) Goal and Scope
- Goal: build a modular tongue diagnosis model for health assessment / disease risk detection using the `shezhenv3-coco` dataset.
- Primary output: a clean, extensible detection pipeline with standardized interfaces.
- Requirements:
  - modular design: `Dataset`, `Backbone`, `DetectionHead`, `Trainer`, `Augments`, `Detector`
  - lowest-cost basic pipeline that runs end-to-end
  - unit tests after each module (small-sample)
  - maintain a framework/tips doc and update on issues
  - add evaluation module after base framework
  - provide performance optimization directions

## 1) Environment Constraints
- Run Python scripts in the `cv` conda environment.
  - Example: `conda run -n cv python train.py`
- Internet access may be blocked; prefer existing environment packages or offline wheels.
- Record issues/tips in `codex.MD` as they occur.

## 2) Current Dataset Stats (shezhenv3-coco)
Collected from local analysis:
- Images (ann/disk): train 5594/5594, val 572/572, test 553/553 (total 6719)
- Annotations: train 14677, val 1874, test 1672 (total 18223)
- Classes: 21 (from `classes.txt`), category order aligned with COCO categories
- Missing classes by split:
  - train: `piweitu`
  - val: `piweitu`, `xinfeitu`
  - test: `shenqutu`, `gandantu`, `piweitu`, `xinfeitu`
- Extreme class imbalance (overall):
  - Top: baitaishe 5040, hongdianshe 3653, liewenshe 1886, chihenshe 1482, hongshe 1466
  - Tail: shenqutu 2, xinfeitu 2, gandantu 9, jiankangshe 21
- Data issue: 7 test annotations use category_id=21 (unknown)

## 2) Architecture Overview (Target)
- `datasets/`: dataset loaders + COCO parser
- `models/`: backbone, neck (optional), detection heads
- `augments/`: augmentation pipeline
- `engine/`: trainer, evaluator
- `configs/`: YAML/JSON configs
- `tools/`: CLI entry points (train, eval, export)
- `docs/FRAMEWORK.md`: overall framework notes and tips

Core interfaces (minimal, standard):
- `Dataset.__len__()`, `Dataset.__getitem__(idx)` -> `(image, target)`
  - `target` uses COCO-style dict: `boxes`, `labels`, `image_id`, `area`, `iscrowd`
- `Backbone.forward(images)` -> `features`
- `DetectionHead.forward(features, targets=None)` -> `losses` (train) / `detections` (eval)
- `Detector.forward(images, targets=None)` -> `losses` or `detections`
- `Trainer.train_one_epoch(...)`, `Trainer.evaluate(...)`
- `Augment.__call__(image, target)` -> `(image, target)`

## 3) Milestones and Module Plan

### M1 - Repo scaffolding + framework docs
Deliverables:
- `docs/FRAMEWORK.md` with architecture, interfaces, and how to run
- `codex.MD` updated with tips/issues (already exists, keep updated)
- folder structure skeleton

Unit tests:
- `tests/test_imports.py` ensures all modules import in `cv` env

### M2 - Dataset module (COCO)
Tasks:
- Implement `datasets/shezhen_coco.py`
  - load `train/val/test` jsons
  - image reading
  - return COCO targets
- Optional: class mapping from `classes.txt`

Unit tests:
- `tests/test_dataset_basic.py`
  - load small subset (e.g., 2 images)
  - validate target keys, box shapes

### M3 - Augments module
Tasks:
- Implement `augments/basic.py` with minimal transforms:
  - resize, random flip, color jitter (if available)
  - ensure boxes are updated correctly
- keep dependency minimal (torchvision transforms preferred)

Unit tests:
- `tests/test_augments.py` with 1-2 images + known bbox

### M4 - Backbone module
Tasks:
- Provide at least one backbone with official pretrained weights: `resnet50` (torchvision)
- Add a lightweight option (e.g., `resnet18` or `mobilenet_v3`)
- Standardize output features shape
- Add `models/backbones/__init__.py`

Unit tests:
- `tests/test_backbone.py` on dummy tensor

### M5 - Detection Head module
Tasks:
- Implement one minimal head (e.g., RetinaNet / Faster R-CNN via torchvision)
- Wrap to match `DetectionHead` interface

Unit tests:
- `tests/test_head.py` on dummy features

### M6 - Detector module
Tasks:
- `models/detector.py` compose backbone + head
- forward in train/eval modes

Unit tests:
- `tests/test_detector.py` for forward pass

### M7 - Trainer module
Tasks:
- `engine/trainer.py` with:
  - train loop
  - optimizer/scheduler
  - checkpointing
  - logging (stdout, simple CSV)

Unit tests:
- `tests/test_trainer_smoke.py` run 1 epoch on 2 images

### M8 - Evaluation module (after base framework)
Tasks:
- `engine/evaluator.py` using COCO metrics
  - AP/AR summary
  - per-class AP
- CLI `tools/eval.py`

Unit tests:
- `tests/test_eval_smoke.py` with tiny subset

### M9 - Imbalance strategy modules (after pipeline runs)
Goal: address class imbalance with pluggable strategies.
Tasks:
- `engine/samplers.py`:
  - oversampling, undersampling, class-aware sampling, targeted sampling
- `models/losses.py`:
  - class-weighted cross entropy
  - focal loss
- `models/anchors.py` or `models/heads/`:
  - anchor strategy tuning
- `augments/prior.py`:
  - tongue prior-based augmentation
  - class-conditional augmentation
Unit tests:
- `tests/test_sampling_strategies.py` (smoke)
- `tests/test_losses.py` (numerical sanity)

### M10 - Comparative experiments
Goal: compare baseline vs each imbalance strategy.
Tasks:
- Define experiment matrix in `configs/`
- Run fixed protocol:
  - same data split, same schedule
  - log metrics (AP, per-class AP, macro-F1)
- Produce a comparison table and short report
Deliverable:
- `docs/IMBALANCE_COMPARISON.md`

### M11 - Performance optimization directions (report)
Deliverables:
- `docs/PERFORMANCE.md` with:
  - data: class imbalance handling, sampling, focal loss
  - model: better backbone, FPN, anchor tuning
  - training: longer schedule, mixed precision, EMA
  - augmentation: scale jitter, mosaic (if needed)
  - inference: TTA, confidence thresholds

## 4) Standardized Interfaces (Details)
- `Dataset`:
  - `image`: torch float32, CHW, [0,1]
  - `boxes`: float32 tensor Nx4 (xmin, ymin, xmax, ymax)
  - `labels`: int64 tensor N
  - `image_id`: int64 scalar

- `Detector`:
  - if `targets` provided -> return `loss_dict`
  - else -> return `detections` list (COCO-style)

## 5) Config + CLI
- Config files in `configs/`:
  - dataset paths, image size, batch size
  - model (backbone, head)
  - optimizer (lr, wd)
  - train schedule
- CLI tools:
  - `tools/train.py`
  - `tools/eval.py`
  - `tools/export.py` (optional)

## 6) Testing Discipline
- After each module, add a small unit test
- Use tiny subset to keep runtime low
- Run in `cv` env:
  - `conda run -n cv pytest -q`

## 7) Documentation Discipline
- Maintain `docs/FRAMEWORK.md` as the single source for architecture
- Update `codex.MD` with tips/issues as they appear

## 8) Acceptance Criteria
- Base pipeline trains 1 epoch on train subset
- Evaluation runs on val subset and outputs AP metrics
- All tests pass in `cv` environment
- Docs are updated and current
- Baseline vs strategy comparisons are documented in `docs/IMBALANCE_COMPARISON.md`

## 9) Handoff Checklist (for next agent)
- Confirm `cv` environment works
- Verify dataset paths
- Run `pytest -q` in `cv`
- Start with M1 and proceed in order
