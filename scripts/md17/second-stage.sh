#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_BASE_URL="https://api.wandb.ai"
export HYDRA_FULL_ERROR=1

ulimit -n 65535

python src/train.py \
    experiment=md17/second-stage \
    trainer.devices=1 \
    model.compile=true \
    data.batch_size=64 \
    trainer.max_epochs=2000 \
    data.num_workers=31 \
    model.backbone.depth=6 \
    -m
