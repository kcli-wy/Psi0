#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# source .venv-dp/bin/activate

python src/dp/deploy/dp_g1_serve_simple.py \
    --host=0.0.0.0 \
    --port=22085 \
    --run-dir=.runs/diffusion-policy-g1/g1wholebodylocomotionpickbetweentablesvariant5-v0-processed.g1.cosine.lr1.0e-04.b64.gpus2.2603030648 \
    --ckpt-step=40000 \
    "$@"
