#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_BASE_URL="https://api.wandb.ai"
export HYDRA_FULL_ERROR=1

ulimit -n 65535

python src/train.py \
    experiment=peptide/second-stage \
    trainer.devices=1 \
    data.batch_size=16 \
    dataset=4AA-sims \
    model.compile=True \
    data.rand_rotation=True \
    data.rand_translation=0.1 \
    trainer.max_epochs=1500 \
    -m
