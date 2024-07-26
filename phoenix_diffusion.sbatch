#!/bin/bash
#SBATCH -J diffusion # Job name
#SBATCH -A gts-jw254-coda20 
#SBATCH -qembers
#SBATCH -N2 --gpus-per-node=RTX_6000:4 # -C A100-80GB              # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=32G                        # Memory per core
#SBATCH -t 10:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

#module load gcc/10.3.0-o57x6h
#module load mvapich2/2.3.6-ouywal

pwd
date
module load anaconda3/2022.05 # Load module dependencies
module load pytorch
module list
conda activate diffusers 
conda env list
cat $0

srun python diffusion.py --train 1 --sample 0 
