<div align="center">

# LaM-SLidE ({La}tent Space {M}odeling of Spatial Dynamical {S}ystems via {Li}nke{d} {E}ntities)

<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

<center>

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/docs/2.5/)
[![lightning](https://img.shields.io/badge/-Lightning_2.4-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</center>

[**Project Page**](https://ml-jku.github.io/LaM-SLidE/) | [**Paper**](https://arxiv.org/abs/2502.12128/)


Implementation of **LaM-SLidE** (Latent Space Modeling of Spatial Dynamical Systems via Linked Entities).

**Note:** This repository is provided for research reproducibility only and is not intended for usage in application workflows.

## News
🔥***February 18, 2025***: *The training code and paper preprint are released.*

## Setup

### Installation

```shell
mamba env create -f environment.yaml
mamba activate pyt25
```

### Env variables

Create an `.env` file and set the parameters for logging with [wandb](https://wandb.ai/). An example can be found [here](.env.example).

## Setup

### Data

The data for all experiment will be located in the `data` directory.

```shell
mkdir data
```

### Workflow

Because our methods reilies on two stage approach:

1. First stage encoder/decoder
2. Second stage latent model

we retrieve [wandb](<%5Bwandb%5D(www.https://wandb.ai)>) first stage model information direclty form the api, this simplyfies the workflow a lot, and for the second stage training we only have to provide the RunID of the first stage to the second stage training.

## Experiments

### MD17

#### Data Preparation

Download the MD17 dataset in `.npz` format from [here](http://www.sgdml.org/#datasets). The dataset should be placed in `data/md17`.

#### Training

```python
# First stage (Encoder-Decoder)
python experiment=md17/first-stage

# Second stage (Diffusion)
python experiment=md17/second-stage first_stage_settings.run_id=[WB_RUN_ID] first_stage_settings.project=[WB_PROJECT]
```

### Pedestrian

#### Data Preparation

Follow the instructions [here](https://github.com/MediaBrain-SJTU/EqMotion?tab=readme-ov-file#data-preparation-3) to download and preprocess the data.
Then move the preprocessed files in the folder `processed_data_diverse` into `data/pedestrian_eqmotion`.

#### Training

```python
# First stage (Encoder-Decoder)
python experiment=pedestrian/first-stage

# Second stage (Diffusion)
python experiment=pedestrian/second-stage first_stage_settings.run_id=[WB_RUN_ID] first_stage_settings.project=[WB_PROJECT]
```

### NBA

#### Data preparation

Download the data from [here](https://github.com/xupei0610/SocialVAE/tree/main/data/nba)

Process the data with following commands.

```python
# Train
python scripts/nba/process_4AA.py --data_dir data/social_vae_data/nba/score/train
python scripts/nba/process_4AA.py --data_dir data/social_vae_data/nba/rebound/train

# Val
python scripts/nba/process_data.py --data_dir data/social_vae_data/nba/score/val
python scripts/nba/process_4AA.py --data_dir data/social_vae_data/nba/rebound/val

```

#### Training

```python
# First stage (Encoder-Decoder)
python experiment=nba/first-stage

# Second stage (Diffusion)
python experiment=nba/second-stage first_stage_settings.run_id=[WB_RUN_ID] first_stage_settings.project=[WB_PROJECT]
```

### Tetrapeptide - 4AA

Follow the instructions [here](https://github.com/bjing2016/mdgen) to download the data.

#### Data preparation

Process the data with the following commands.

```python
# Train
python scripts/peptide/process_4AA.py --split data/mdgen/splits/4AA_train.csv --outdir data/mdgen/4AA_sims_processed/train --sim_dir data/mdgen/4AA_sims

# Val
python scripts/peptide/process_4AA.py --split data/mdgen/splits/4AA_val.csv --outdir data/mdgen/4AA_sims_processed/val --sim_dir data/mdgen/4AA_sims

# Test
python scripts/peptide/process_4AA.py --split data/mdgen/splits/4AA_test.csv --outdir data/mdgen/4AA_sims_processed/test --sim_dir data/mdgen/4AA_sims
```

#### Training

```python
# First stage (Encoder-Decoder)
python experiment=peptide/first-stage

# Second stage (Diffusion)
python experiment=peptide/second-stage first_stage_settings.run_id=[WB_RUN_ID] first_stage_settings.project=[WB_PROJECT]
```

# Acknowledgments

Our source code was inpired by previous work:

- [mdgen](https://github.com/bjing2016/mdgen) - Latent space conditioning/masking.
- [flux](https://github.com/black-forest-labs/flux) - Latent space model architecture.
- [SiT](https://github.com/willisma/SiT) - Stochastic interpolants framework.
- [UPT](https://github.com/ml-jku/UPT/) - Encoder - decoder architecture.

# Citation

If you like our work, please consider giving it a star 🌟 and cite us

```
@misc{sestak2025lamslidelatentspacemodeling,
      title={LaM-SLidE: Latent Space Modeling of Spatial Dynamical Systems via Linked Entities}, 
      author={Florian Sestak and Artur Toshev and Andreas Fürst and Günter Klambauer and Andreas Mayr and Johannes Brandstetter},
      year={2025},
      eprint={2502.12128},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12128}, 
}
```