# ml21cm

Conditional diffusion models for two- and three-dimensional cosmological
21 cm lightcone emulation.

This repository contains the data-generation, preprocessing, training,
sampling, and evaluation code used in:

**Three-dimensional Conditional Diffusion Models for Cosmological 21 cm Lightcone Emulation** (arXiv:2605.29016)
## Overview

The models generate 21 cm brightness-temperature lightcones conditioned on

- minimum halo virial temperature, $\log_{10} T_{\rm vir}$\;
- ionizing efficiency, $\zeta$.

The code supports

- 2D transverse--line-of-sight lightcones;
- full 3D lightcone volumes;
- conditional DDPM training and sampling;
- Yeo--Johnson, min--max, z-score, and arcsinh preprocessing;
- optional transformed-space amplitude compression;
- evaluation using brightness-temperature images, global signals, power
  spectra, voxel PDFs, and scattering coefficients.

The simulations used in the paper were generated with
`py21cmfast==3.3.1`, with varying initial conditions and
spin-temperature fluctuations enabled.

## Installation

Create a Python environment and install the required packages, including:

```text
numpy
scipy
h5py
matplotlib
torch
mpi4py
py21cmfast==3.3.1
kymatio
````

The large-scale experiments reported in the paper were run with distributed
PyTorch on NVIDIA A100 GPUs.

## Usage

The main workflow is:

1. Generate 21cmFAST lightcones and store them in HDF5 format.
2. Train a conditional 2D or 3D diffusion model.
3. Sample lightcones at selected astrophysical parameter values.
4. Evaluate the generated ensembles with the supplied physical diagnostics.

Training is implemented in:

```text
training/diffusion.py
```

Available command-line options can be inspected with:

```bash
python training/diffusion.py --help
```

The paper's preferred models use Yeo--Johnson preprocessing followed by a
linear amplitude compression with (A=0.1).

## Data

The complete training and validation datasets are not included in this
repository because of their size. They are available from the corresponding
author upon reasonable request.

Each generated HDF5 file records the simulation fields, astrophysical
parameters, random seeds, redshifts, distances, and generation settings.

## Citation

A frozen version of the software associated with the paper is archived on
Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21402795.svg)](https://doi.org/10.5281/zenodo.21402795)

Please cite the archived Zenodo release rather than only the continuously
updated GitHub repository.

## License

This software is released under the [MIT License](LICENSE).

