#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_BASE_URL="https://api.wandb.ai"
export HYDRA_FULL_ERROR=1

ulimit -n 65535

python src/train.py \
    experiment=nba/second-stage \
    model.compile=True \
    data.batch_size=1024 \
    trainer.devices=1 \
    trainer.max_epochs=1000 \
    -m
