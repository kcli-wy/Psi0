#!/usr/bin/env bash
set -euo pipefail

source .env
source ./src/gr00t/.venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf_modules_gr00t}"
mkdir -p "$HF_MODULES_CACHE"
NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-29527}"

G1_DATASET_PATH="${G1_DATASET_PATH:-/hfm/data/HE_RAW_no_static_lerobot_g1}"
H1_DATASET_PATH="${H1_DATASET_PATH:-/hfm/data/HE_RAW_no_static_lerobot_h1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/pretrain_he_g1_h1_mixed_scratch_gr00t}"
TUNE_TOP_LLM_LAYERS="${TUNE_TOP_LLM_LAYERS:-4}"

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  baselines/gr00t-n1.6/launch_train_he_mixed_ee.py \
  --g1-dataset-path "$G1_DATASET_PATH" \
  --h1-dataset-path "$H1_DATASET_PATH" \
  --base-model-path "$BASE_MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --tune-top-llm-layers "$TUNE_TOP_LLM_LAYERS" \
  --scratch-gr00t \
  --override-pretraining-statistics
