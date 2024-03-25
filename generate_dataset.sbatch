#!/bin/bash
#SBATCH -Jgenerate_dataset                    # Job name
#SBATCH --account=gts-jw254-coda20                 # charge account
#SBATCH -qembers                               # QOS Name
#SBATCH -N4 -c22                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=32G                        # Memory per core
#SBATCH -t8:00:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

#module load gcc/10.3.0-o57x6h
#module load mvapich2/2.3.6-ouywal
module load anaconda3/2022.05 # Load module dependencies
conda activate 21cmFAST
srun python generate_dataset.py
