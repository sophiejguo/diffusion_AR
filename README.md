# diffusion_AR

**Diffusion-Augmented Autoregressive Image Generation via Next-Scale Prediction**
*Sophie Guo · Sandeep Kambhampati*

This is the code repository accompanying the paper [“Diffusion-Augmented Autoregressive Image Generation via Next-Scale Prediction”](final-report.pdf).

## Background and Motivation

State-of-the-art autoregressive (AR) image generation models, such as [Visual Autoregressive Modeling (VAR)](https://arxiv.org/abs/2404.02905), achieve scalability through multi-scale coarse-to-fine prediction. However, they rely on **vector quantization (VQ)** to discretize continuous latent representations into discrete codebook tokens before applying a cross-entropy training objective. While effective, VQ introduces three fundamental limitations:

1. **Information loss** — quantization discards high-frequency details, leading to reconstruction artifacts and blurred outputs.
2. **Codebook collapse** — training instability arises when large portions of the codebook remain unused.
3. **Scalability–expressivity tradeoff** — larger codebooks improve fidelity but increase memory costs and complicate sequence modeling.

## What This Project Proposes

We propose **DiffusionVAR**, which eliminates the VQ bottleneck by replacing discrete token prediction with a **continuous diffusion objective** over the raw encoder residuals at each scale. Rather than predicting codebook indices via cross-entropy, a small per-token denoising MLP (conditioned on the AR transformer’s hidden state) is trained via DDPM ε-prediction loss on a cosine noise schedule. This decouples autoregressiveness from discretization: the AR mechanism remains intact, but each “token” becomes a continuous latent vector whose distribution is modeled via diffusion.

Key design choices (see `final-report.pdf`, Sections 2–3):

- The diffusion targets are the **pre-quantization continuous residuals** at each scale — the raw `quant_conv(encoder(img))` output before the nearest-neighbour codebook lookup — not the quantized embeddings.
- The VQ codebook is still used to produce **discrete token indices for teacher-forced AR context**, keeping the transformer backbone compatible with pretrained VAR weights.
- **No classifier-free guidance** is applied at inference; generation uses iterative DDPM denoising (`diffusion_sample`).
- **Teacher forcing** (using ground-truth scale tokens as context for subsequent scales) is used during training and measurably improves generation quality.

## Experiments and Results

Experiments were conducted on CIFAR-10 (training from scratch) and ImageNet (fine-tuning a pretrained VAR depth=12 on the first ten classes). Key findings:

- A non-quantized multi-scale autoencoder reconstructs images with virtually no perceptual degradation compared to the VQ counterpart, motivating continuous token modeling.
- Codebook size ablations (V=1024 vs V=4096) on CIFAR-10 VAR show similar FID/IS, suggesting the quantization bottleneck is not simply resolved by scaling the codebook.
- Fine-tuning with diffusion loss produced generations with reasonable structure and class consistency, but did not clearly outperform the cross-entropy fine-tuning baseline. The authors hypothesize that fine-tuning from a CE-pretrained checkpoint is challenging due to objective misalignment, and that training DiffusionVAR from scratch may yield better results.

See [`final-report.pdf`](final-report.pdf) for full experimental details, figures, and discussion of related work (FlowAR, xAR).

---

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

**VAR (discrete):**
```text
image
  -> VQ-VAE encoder + quant_conv
  -> VectorQuantizer: nearest-neighbour lookup at each scale -> discrete token indices
  -> VAR transformer: predicts next scale's token distribution (cross-entropy logits)
  -> VQ-VAE decoder
  -> generated image
```

**DiffusionVAR (continuous):**
```text
image
  -> VQ-VAE encoder + quant_conv
  -> residual quantization loop -> pre-quantization continuous residuals z_s at each scale
                                -> discrete token indices (for teacher-forced AR context only)
  -> DiffusionVAR transformer: outputs conditioning vector phi_s per token
  -> DiffLoss MLP: denoises z_s conditioned on phi_s (DDPM, cosine schedule, 1000 steps)
  -> VQ-VAE decoder
  -> generated image
```

The key difference: DiffusionVAR trains the diffusion denoiser to reconstruct the raw encoder
residual at each scale *before* the nearest-neighbour codebook lookup, bypassing the VQ
information bottleneck for the generative objective. The VQ codebook is still used to
provide teacher-forced context to the AR transformer backbone.

More concretely:

- `models/vqvae.py` encodes and decodes images; `img_to_idxBl_and_prequant` extracts both
  discrete indices and pre-quantization continuous residuals in a single pass
- `models/quant.py` implements multi-scale residual vector quantization; `f_to_idxBl_and_prequant`
  returns per-scale continuous residuals alongside discrete indices
- `models/var.py` predicts discrete multi-scale token indices via cross-entropy
- `models/diffusion_var.py` outputs continuous token embeddings for conditioning `DiffLoss`;
  inference uses `diffusion_sample` (iterative DDPM denoising, no classifier-free guidance)
- `models/diffloss.py` implements the per-token DDPM denoiser (small AdaLN MLP, cosine schedule)
- `trainers/var_trainer.py` and `trainers/diffusion_var_trainer.py` handle optimization,
  evaluation, and checkpoint saving/loading

## Data

The training code assumes an ImageNet-style CIFAR-10 directory structure or a similar image-folder layout. The default data path is defined in `utils/arg_util.py` and can be overridden with CLI flags.

If you need a custom dataset layout, update:

- `configs/data/imagenet_style_cifar10.yaml`
- `utils/data.py`

The training scripts also expect a pretrained VQ-VAE checkpoint in some flows:

- `scripts/train_var.py` and `scripts/train_diffloss.py` auto-download the VQ-VAE checkpoint
  (`vae_ch160v4096z32.pth`) from HuggingFace Hub on first run.
- `scripts/train.py` uses a locally trained taming-transformers checkpoint. If that file is
  not found, the script raises a `FileNotFoundError` with guidance rather than attempting a
  download. You must train the taming-transformers VQ-VAE separately or adjust the path in the script.

## Training

The canonical training entrypoints are in `scripts/`:

| Script | Model | VQ-VAE | Notes |
|---|---|---|---|
| `scripts/train.py` | VAR (discrete) | taming-transformers (local, V=1024, ch=128) | must supply checkpoint locally |
| `scripts/train_var.py` | VAR (discrete) | HuggingFace (V=4096, ch=160) | auto-downloads checkpoint |
| `scripts/train_diffloss.py` | DiffusionVAR (continuous) | HuggingFace (V=4096, ch=160) | auto-downloads checkpoint |

```bash
python -m scripts.train_var
python -m scripts.train_diffloss
```

For `train_diffloss.py`, you can optionally initialize the DiffusionVAR transformer backbone
from a pretrained VAR checkpoint (fine-tuning scenario from Section 3 of the report):

```bash
python -m scripts.train_diffloss --use_pretrained
```

These scripts handle:

- distributed initialization
- dataset loading
- VAE/VAR/DiffusionVAR model construction via `build_vae_var` / `build_vae_diffusion_var`
- checkpoint loading and saving
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

Several scripts expect pretrained checkpoints for the shared VQ-VAE backbone.

| Script | Checkpoint | Behavior if missing |
|---|---|---|
| `scripts/train.py` | taming-transformers VQ-VAE (local) | raises `FileNotFoundError` |
| `scripts/train_var.py` | `vae_ch160v4096z32.pth` | auto-downloaded from HuggingFace Hub |
| `scripts/train_diffloss.py` | `vae_ch160v4096z32.pth` | auto-downloaded from HuggingFace Hub |
| `demos/sample.py` | `vae_ch160v4096z32.pth` + VAR weights | auto-downloaded from HuggingFace Hub |

When `--use_pretrained` is passed to `train_diffloss.py`, the script also downloads a
pretrained VAR backbone checkpoint (`var_d{depth}.pth`) from HuggingFace Hub to initialize
the DiffusionVAR transformer, then loads it with `strict=False` (extra diffusion heads
are randomly initialized).

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
- The codebase is research-oriented and may require adjusting paths, checkpoints, and hyperparameters before a run succeeds.
- The `scripts/` modules are the main executable entrypoints now.
- DiffusionVAR does **not** support classifier-free guidance via logit interpolation (there are
  no discrete logits). Inference always uses iterative DDPM denoising via `diffusion_sample`.
- The diffusion loss targets are the **pre-quantization** continuous residuals at each scale
  (output of `quant_conv(encoder(img))` before the nearest-neighbour codebook lookup), not
  the quantized codebook embeddings. This is the key design choice that removes the VQ
  information bottleneck from the generative path while keeping the VQ codebook for AR context.
