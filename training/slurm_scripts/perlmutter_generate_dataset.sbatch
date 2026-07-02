#!/bin/bash
#SBATCH -J dataset
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N2
#SBATCH --ntasks-per-node=1
#SBATCH -t 15:00:00
#SBATCH -o training/logs/%x-%j
#SBATCH --mail-type=BEGIN,END,FAIL

#----------------------------------------------------
mkdir -p training/logs
date
cat $0
echo "$(printf '%0.s🚥 ' {1..30})"


module load PrgEnv-gnu
module load cray-mpich
module load python

conda activate 21cmFAST

which python
conda env list

srun python generate_dataset.py \
    --save_direc $SCRATCH \
    --num_images 800\
    --BOX_LEN 128 \
    --HII_DIM 64 \
    --NON_CUBIC_FACTOR 16 \
    --cpus_per_node 100 \
#----------------------------------------------------

date
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Start,End,Elapsed

