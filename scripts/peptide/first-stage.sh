#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_BASE_URL="https://api.wandb.ai"
export HYDRA_FULL_ERROR=1

ulimit -n 65535

python src/train.py \
    experiment=peptide/first-stage \
    trainer.devices=1 \
    data.batch_size=512 \
    trainer.max_epochs=50000 \
    data.n_jobs=50 \
    model.compile=True \
    -m
