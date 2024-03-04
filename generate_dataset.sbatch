#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH --account=gts-jw254-coda20                 # charge account
#SBATCH -N4 -c3                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=24G                        # Memory per core
#SBATCH -t10:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qembers                               # QOS Name
#SBATCH -oReport-%j                         # Combined output and error messages file
#SBATCH --mail-type=END,FAIL              # Mail preferences

#cd scratch                            # Change to working directory

module load gcc/10.3.0-o57x6h
module load mvapich2/2.3.6-ouywal
#module load py-mpi4py/3.1.2-mva2-rzdjbn
module load anaconda3/2022.05 # Load module dependencies
conda activate 21cmFAST
#conda activate testenv
#srun python -c "print('python starts.')"
#srun python test_mpi.py #generate_dataset.py                             # Example Process
srun python generate_dataset.py
#srun python -c "print('python ends.')"
