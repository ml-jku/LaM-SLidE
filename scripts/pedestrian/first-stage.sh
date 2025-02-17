#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_BASE_URL="https://api.wandb.ai"
export HYDRA_FULL_ERROR=1

ulimit -n 65535

python src/train.py \
    experiment=pedestrian/first-stage \
    sweep=pedestrian/all \
    n_gpus=1 \
    data.batch_size=256 \
    model.compile=True \
    tags=["final"] \
    -m
