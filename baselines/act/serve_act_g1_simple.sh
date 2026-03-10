#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# source .venv-act/bin/activate

python src/act/deploy/act_g1_serve_simple.py \
    --host=0.0.0.0 \
    --port=22085 \
    --run-dir=.runs/act-g1/g1wholebodylocomotionpickbetweentablesvariant5-v0-processed.g1.cosine.lr1.0e-04.b128.gpus4.2603020706 \
    --ckpt-step=40000 \
    "$@"
