#!/usr/bin/env bash
set -euo pipefail

source ~/.env
source ./src/gr00t/.venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-29501}"

export DATASET_PATH="${DATASET_PATH:-/hfm/data/simple/simple/G1WholebodyBendPick-v0-psi0}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/pretrained_g1_ee_downstream}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"

# Ensure modality config can read dataset meta to set correct action dim.
export SIMPLE_DATASET_PATH="$DATASET_PATH"

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  src/gr00t/gr00t/experiment/launch_finetune.py \
  --base-model-path $PRETRAINED_MODEL_PATH\
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag G1_LOCO_DOWNSTREAM \
  --modality-config-path src/gr00t/gr00t/configs/modality/g1_locomanip.py \
  --num-gpus 3 \
  --output-dir "$OUTPUT_DIR" \
  --save-steps 10000\
  --save-total-limit 4 \
  --max-steps 50000 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --learning-rate 1e-4 \
  --global-batch-size 24 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4
