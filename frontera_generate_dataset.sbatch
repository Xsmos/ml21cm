#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** MPI Job in Normal Queue ***
# 
# Last revised: 20 May 2019
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch clx.mpi.slurm" on a Frontera login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per CLX node: 56
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J datasets           # Job name
#SBATCH -o Report-%j       # Name of stdout output file
#SBATCH -p small # Queue (partition) name
#SBATCH -N 2 # 50              # Total # of nodes 
#SBATCH -t 09:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=xiabin@gatech.edu
#SBATCH --ntasks-per-node=1

# Any other commands must follow all #SBATCH directives...
############# #SBATCH -c 56              # Total # of mpi tasks

#----------------------------------------------------
cat $0
date
pwd
module list
conda env list

srun python generate_dataset.py \
    --save_direc $SCRATCH \
    --num_images 800\
    --BOX_LEN 128 \
    --HII_DIM 64 \
    --NON_CUBIC_FACTOR 16 \
    --cpus_per_node 38 \
#----------------------------------------------------

