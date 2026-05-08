# Synthetic CAPTCHA — cGAN vs Conditional Diffusion

Final project for Udacity's *Generative AI / Diffusion* nanodegree (nd101).
The task is to generate synthetic handwriting that could replace real
human-collected CAPTCHA data, using two complementary generative models
and comparing their quality.

## Notebooks

| Notebook | Phase |
|---|---|
| `00_data_preparation.ipynb` | Load MNIST, normalise to `[-1, 1]`, save a 100-image sample batch |
| `01_cGAN_training.ipynb`     | Conditional GAN — Generator + Discriminator + adversarial loss + checkpointing |
| `02_diffusion_training.ipynb` | Conditional UNet diffusion (DDPM) with timestep + label embeddings |
| `03_evaluation.ipynb`        | Visual + FID + downstream-classifier comparison of the two models |

## Architectures

* **cGAN** (`model/cgan.py`) — MLP Generator with `Embedding(10)` label
  conditioning concatenated with the latent z, BatchNorm + ReLU, tanh
  output. Discriminator mirrors the structure with LeakyReLU + sigmoid.
* **Conditional UNet** (`model/diffusion.py`) — small UNet with
  `GroupNorm`, `ResidualBlock`s, sinusoidal timestep embedding +
  learned label embedding injected into every block.

## Running

```bash
pip install torch torchvision torchmetrics tqdm matplotlib
jupyter notebook starter/00_data_preparation.ipynb
# Then run 01, 02, 03 in order.
```

> **GPU recommended.** On CPU / MPS the cGAN trains in ~30 minutes for
> 50 epochs but the diffusion model takes 2–3 hours per 40 epochs.
> Both notebooks save checkpoints to `../checkpoints/` so the
> evaluation notebook can be re-run without re-training.

## What I implemented

* `01_cGAN_training` — extracted `train_loader` from
  `get_mnist_loaders`, instantiated G/D + Adam optimizers
  (`betas=(0.5, 0.999)`) + `BCELoss`, and ran the
  `train_cgan` helper in 10-epoch blocks with intermediate
  `save_checkpoint` calls.
* `02_diffusion_training` — instantiated `ConditionalUNet`,
  ran `train_diffusion` for 40 epochs, saved the final state.
* `03_evaluation` — loaded both checkpoints, generated samples per
  class, plotted real / cGAN / diffusion side-by-side, computed
  FID against a subsample of real MNIST for both models, and ran
  a small `SimpleCNN` classifier trained on synthetic data
  evaluated on real MNIST (the "utility" metric).

## Standing-out work

* **Per-class one-image baseline** in evaluation —
  `get_real_examples_per_class` returns one image per digit
  deterministically, so the FID baseline is reproducible.
* **Sample normalisation** before the downstream classifier
  (`normalize_synthetic`) so cGAN and diffusion outputs are on the
  same `[-1, 1]` scale as real MNIST.
* **Checkpoint cadence** — cGAN saves every 10 epochs, not just at
  the end, so a crashed training run keeps the latest snapshot.

## License

Educational submission for Udacity nd101.
