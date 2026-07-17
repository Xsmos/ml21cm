# ml21cm

Conditional 2D and 3D diffusion models for emulating cosmological 21 cm
brightness-temperature lightcones.

This repository accompanies **Three-dimensional Conditional Diffusion Models
for Cosmological 21 cm Lightcone Emulation** by Bin Xia and John H. Wise
([arXiv:2605.29016](https://arxiv.org/abs/2605.29016)). It contains the code used
to generate 21cmFAST simulations, train and sample the conditional DDPMs, and
evaluate the resulting ensembles.

## What the model does

The emulator generates differential 21 cm brightness-temperature lightcones
conditioned on two global astrophysical parameters:

- minimum halo virial temperature, $\log_{10}(T_{\rm vir}/{\rm K}) \in [4,6]$;
- ionizing efficiency, $\zeta \in [10,250]$.

The shared U-Net implementation supports both a 2D transverse--line-of-sight
representation, $H\times Z$, and a contiguous 3D volume, $H\times H\times Z$.
It uses 2D or 3D convolutions according to the supplied stride dimensions and
injects the diffusion timestep and the two conditioning parameters into every
residual block.

The paper studies lightcones with $H=64$ and line-of-sight depths up to
$Z=1024$. Its main training set contains 25,600 py21cmfast v3.3.1 lightcones,
with one realization per sampled parameter pair. Validation uses five fixed
parameter points with 800 independent realizations at each point. The reported
comparison uses the full 1,000-step DDPM sampling trajectory and evaluates
images, global signals, power spectra, voxel PDFs, and reduced scattering
coefficients.

The main empirical result is that preprocessing matters more than the other
tested hyperparameters. Among the configurations studied, a Yeo--Johnson power
transform followed by moderate linear amplitude compression (`--squish 0.1 0`)
provides the most reliable trade-off. The matched 2D model performs better on
the global signal and reduced scattering coefficients, while the 3D model has
smaller central-residual errors in the power spectrum.

## Repository layout

```text
models/context_unet.py              dimension-agnostic conditional U-Net
training/diffusion.py               distributed DDPM training and sampling
training/slurm_scripts/             example Slurm launch scripts
utils/generate_dataset.py           MPI-enabled py21cmfast data generation
utils/load_h5.py                    HDF5 loading and preprocessing
utils/PowerTransformer_*.pkl        fitted Yeo--Johnson transformers
utils/plot3D.py                     3D volume rendering
utils/plot_mp4.py                   animation helper
evaluation/evaluate.ipynb           ensemble diagnostics and paper figures
evaluation/plot_global_signal4hyperparameters.py
                                    global-signal hyperparameter comparison
```

## Requirements

The code targets Linux HPC systems with Slurm, MPI, NVIDIA GPUs, and distributed
PyTorch. The paper's full 3D runs used 32 NVIDIA A100 80 GB GPUs; a
$64\times64\times1024$ volume still required a micro-batch size of 2, gradient
accumulation, mixed precision, and activation checkpointing.

Create an environment with a CUDA-compatible PyTorch build and install the
runtime dependencies:

```bash
python -m pip install numpy scipy h5py matplotlib tqdm psutil \
    scikit-learn joblib huggingface-hub wandb mpi4py py21cmfast==3.3.1
```

The evaluation and visualization utilities additionally use `kymatio`,
`Pillow`, `vedo`, and a system `ffmpeg` installation. Exact package versions are
not currently pinned, so record your environment when running a reproducibility
study.

## Data generation

Generate lightcones with MPI through `utils/generate_dataset.py`. For example:

```bash
srun python utils/generate_dataset.py \
    --save_direc /path/to/data \
    --num_images 800 \
    --BOX_LEN 128 \
    --HII_DIM 64 \
    --NON_CUBIC_FACTOR 16 \
    --cpus_per_node 100
```

This example produces brightness temperature, density, and neutral-fraction
fields in HDF5. The file also records parameter keys and values, random seeds,
redshifts, distances, and simulation settings.

Before launching a new parameter-space campaign, edit `params_ranges` near the
bottom of `utils/generate_dataset.py`. The checked-in default is the single
validation point $(\log_{10}T_{\rm vir},\zeta)=(4.8,131.341)$; use `[4, 6]` and
`[10, 250]` to sample the training domain. The generator enables spin-temperature
fluctuations and varies the initial-condition seed between realizations.

The complete paper datasets are not stored in this repository because of their
size. They are available from the corresponding author upon reasonable request.

## Training

`training/diffusion.py` is a distributed entry point. Launch it with `torchrun`
under Slurm so that `MASTER_ADDR`, `MASTER_PORT`, `SLURM_NNODES`, and the rank
variables are defined. The supplied `perlmutter_diffusion.sbatch` reproduces the
main 120-epoch 3D configuration:

```bash
sbatch perlmutter_diffusion.sbatch
```

Update the dataset path and Slurm allocation directives before submitting. Its
effective model command is:

```bash
torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    training/diffusion.py \
    --train /path/to/lightcones.h5 \
    --num_image 800 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_redshift 1024 \
    --channel_mult 1 1 2 4 \
    --stride 2 2 4 \
    --num_res_blocks 1 \
    --n_epoch 120 \
    --lrate 1e-5 \
    --scale_path utils/PowerTransformer_25600_z1.pkl \
    --squish 0.1 0
```

`--num_image` is the number of lightcones loaded by each GPU, not the global
dataset size. The number of stride values selects the model dimensionality:
use two values for a 2D model and three for a 3D model. Checkpoints and samples
are written to `training/outputs/`, and experiment tracking uses Weights &
Biases. Run `python training/diffusion.py --help` to list all command-line
options; note that even `--help` currently requires the Python dependencies to
be installed.

## Sampling and evaluation

Resume a checkpoint and enable sampling with `--sample 1`:

```bash
# Add these options to the distributed training command above.
--resume training/outputs/model-...-epoch120.pt \
--sample 1 \
--num_new_img_per_gpu 50 \
--max_num_img_per_gpu 2
```

Sampling conditions are currently defined in `params_pairs` near the end of
`training/diffusion.py`; edit that list for the desired
$(\log_{10}T_{\rm vir},\zeta)$ values. Use the same architecture,
preprocessing, and amplitude-compression arguments as the checkpoint. The
`--entire 1` option assembles a full output volume when applicable.

Use `evaluation/evaluate.ipynb` to compare generated and 21cmFAST ensembles.
It contains the image, global-signal, power-spectrum, voxel-PDF, and scattering
coefficient analyses used in the paper. Several notebook paths and checkpoint
names are experiment-specific and must be changed for a new environment.

## Reproducibility notes

- The published validation points are `(4.4, 131.341)`, `(5.477, 200)`,
  `(4.699, 30)`, `(5.6, 19.037)`, and `(4.8, 131.341)`.
- All five points are inside the training range; the paper evaluates
  interpolation, not extrapolation.
- The preferred 2D and 3D models use Yeo--Johnson preprocessing,
  `--squish 0.1 0`, one encoder residual block per level, and 240 epochs.
- The reported DDPM uses 1,000 diffusion steps, linear betas from $10^{-4}$ to
  $0.02$, AdamW, cosine learning-rate decay, MSE noise prediction, and gradient
  clipping at 1.0.
- EMA with decay 0.995 was tested but did not improve the reported results.
- The code and scripts retain machine-specific paths from the original HPC
  experiments; inspect them before launching a job.

## Citation

A frozen software release is archived on Zenodo. If you use this repository,
please cite that release and the accompanying paper:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21402795.svg)](https://doi.org/10.5281/zenodo.21402795)

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
