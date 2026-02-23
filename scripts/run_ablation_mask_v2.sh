#!/bin/bash
# Mask增强与多任务学习对比实验 - 手动执行脚本
# 使用方法: conda activate cv && bash scripts/run_ablation_mask_v2.sh

set -e

OUTPUT_BASE="runs/ablation_mask_v2"
DATA_ROOT="datasets/shezhenv3-coco"

# 公共参数
COMMON_ARGS="--data-root $DATA_ROOT \
    --image-size 640 \
    --batch-size 16 \
    --backbone resnet50 \
    --epochs 50 \
    --lr 0.0001 \
    --loss focal \
    --early-stop 10 \
    --amp"

echo "=============================================="
echo "Mask增强与多任务学习对比实验"
echo "=============================================="
echo ""

# 实验1: B0_baseline (已完成，跳过)
echo "[1/4] B0_baseline - 已完成，跳过"
# python scripts/train_classifier.py $COMMON_ARGS \
#     --model-type baseline \
#     --output-dir $OUTPUT_BASE/B0_baseline

# 实验2: A1_mask_crop
echo ""
echo "[2/4] A1_mask_crop - Mask裁剪增强"
echo "=============================================="
python scripts/train_classifier.py $COMMON_ARGS \
    --model-type baseline \
    --mask-aug \
    --mask-aug-mode crop \
    --output-dir $OUTPUT_BASE/A1_mask_crop

# 实验3: A2_mask_bg_blur (已完成，跳过)
echo ""
echo "[3/4] A2_mask_bg_blur - 已完成，跳过"
# python scripts/train_classifier.py $COMMON_ARGS \
#     --model-type baseline \
#     --mask-aug \
#     --mask-aug-mode background \
#     --mask-aug-bg-mode blur \
#     --mask-aug-dilate 25 \
#     --output-dir $OUTPUT_BASE/A2_mask_bg_blur

# 实验4: M1_multitask_seg_v2
echo ""
echo "[4/4] M1_multitask_seg_v2 - 分类+分割联合训练"
echo "=============================================="
python scripts/train_classifier.py $COMMON_ARGS \
    --model-type seg_attention_v2 \
    --seg-loss bce_dice \
    --seg-loss-weight 0.2 \
    --train-seg \
    --soft-floor 0.1 \
    --output-dir $OUTPUT_BASE/M1_multitask_seg_v2

# 分析结果
echo ""
echo "=============================================="
echo "实验完成，分析结果..."
echo "=============================================="
python scripts/run_mask_aug_multitask_compare.py --analyze --output-base $OUTPUT_BASE

echo ""
echo "Done!"
