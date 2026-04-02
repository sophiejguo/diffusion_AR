# diffusion_AR

This repository contains an experimental image generation stack that compares:

- a multi-scale autoregressive model (`VAR`)
- a diffusion-augmented autoregressive variant (`DiffusionVAR`)
- a shared multi-scale VQ-VAE backbone for tokenization and reconstruction

The code has been reorganized so the canonical implementations live in:

- `models/`
- `trainers/`
- `utils/`
- `scripts/`
- `demos/`

The repo root keeps only a small number of compatibility shims for legacy imports.

## What’s here

- `models/`: VQ-VAE, VAR, diffusion VAR, and supporting blocks
- `trainers/`: training/eval loops for the AR and diffusion variants
- `utils/`: distributed setup, logging, datasets, AMP, EMA, and helpers
- `scripts/`: executable training entrypoints
- `demos/`: sampling and editing demos
- `configs/`: YAML config stubs and defaults
- `taming-transformers/`: vendored upstream components and assets used by the VQ-VAE stack
- `jobs/`: sample job scripts
- `legacy/`: temporary compatibility folders kept as placeholders

## Main ideas

The project explores a coarse-to-fine image generation pipeline:

1. Train a multi-scale VQ-VAE to encode images into latent tokens.
2. Train an autoregressive model to predict those tokens scale by scale.
3. Replace or augment discrete token prediction with a diffusion objective for continuous latent prediction.

In practice, this lets you compare reconstruction quality, training stability, and generation quality between:

- discrete multi-scale token generation
- continuous embedding generation with diffusion loss

## Installation

Create a Python environment and install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

The code expects PyTorch and distributed training support. The commented PyTorch line in `requirements.txt` is a reminder to choose a build that matches your CUDA stack.

Recommended baseline:

- Python 3.10 or 3.11
- PyTorch 2.1.x or 2.2.x
- CUDA 11.8 or CUDA 12.x
- Linux is strongly recommended for distributed training and optional fused kernels

Some optional features also use:

- `huggingface_hub`
- `typed-argument-parser`
- `torchvision`
- `wandb`
- FlashAttention or xFormers, if available

If you plan to use the optional fused attention or MLP paths, install the matching extensions for your PyTorch/CUDA build. The code falls back to standard PyTorch operators when those packages are unavailable.

## Quick Start

From a fresh checkout:

```bash
python -m pip install -r requirements.txt
python -m scripts.train --help
```

Train the main VAR model:

```bash
python -m scripts.train
```

Train the diffusion-augmented variant:

```bash
python -m scripts.train_diffloss
```

Run the sampling demo:

```bash
python -m demos.sample
```

If you want the notebook version instead, open:

- `demos/notebooks/demo_sample.ipynb`
- `demos/notebooks/demo_zero_shot_edit.ipynb`

## Model Flow

```text
image
  -> multi-scale VQ-VAE encoder
  -> latent token maps / embeddings
  -> VAR or DiffusionVAR
  -> autoregressive or diffusion-based prediction across scales
  -> VQ-VAE decoder
  -> reconstructed / generated image
```

More concretely:

- `models/vqvae.py` encodes and decodes images
- `models/var.py` predicts discrete multi-scale token indices
- `models/diffusion_var.py` predicts continuous embeddings with a diffusion objective
- `trainers/var_trainer.py` and `trainers/diffusion_var_trainer.py` handle optimization, evaluation, and progressive scale scheduling

## Data

The training code assumes an ImageNet-style CIFAR-10 directory structure or a similar image-folder layout. The default data path is defined in `utils/arg_util.py` and can be overridden with CLI flags.

If you need a custom dataset layout, update:

- `configs/data/imagenet_style_cifar10.yaml`
- `utils/data.py`

The training scripts also expect a pretrained VQ-VAE checkpoint in some flows. If the file is not present, the scripts may try to download it or use a local path defined in the code.

## Training

The canonical training entrypoints are in `scripts/`:

```bash
python -m scripts.train
python -m scripts.train_var
python -m scripts.train_diffloss
```

These scripts handle:

- distributed initialization
- dataset loading
- VAE/VAR model construction
- checkpoint loading
- logging and evaluation

The root-level training files were removed during cleanup, so the `scripts/` modules are the primary interface now.

## Demos

Sampling and editing demos live under `demos/`.

```bash
python -m demos.sample
python -m demos.zero_shot_edit
```

The demo notebooks are stored in:

- `demos/notebooks/demo_sample.ipynb`
- `demos/notebooks/demo_zero_shot_edit.ipynb`

## Checkpoints

Several scripts expect pretrained checkpoints, especially for the shared VQ-VAE backbone. If a checkpoint is not found locally, the training or demo code may try to download it or point to a local path.

Common checkpoint references appear in:

- `scripts/train.py`
- `scripts/train_var.py`
- `scripts/train_diffloss.py`
- `demos/sample.py`

## Repository structure

```text
diffusion_AR/
  configs/
  demos/
  diffusion_utils/
  jobs/
  legacy/
  models/
  scripts/
  taming-transformers/
  trainers/
  utils/
  README.md
  LICENSE
  requirements.txt
```

## Notes

- The repo still includes compatibility shims such as `dist.py` so older import paths continue to work.
- Some files and configs are still placeholders or migration stubs.
- The codebase is research-oriented and may require adjusting paths, checkpoints, and hyperparameters before a run succeeds.
- The `scripts/` modules are the main executable entrypoints now.
