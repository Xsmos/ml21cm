#!/bin/bash
#SBATCH -A m4717 
#SBATCH -J diffusers
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N1 
#SBATCH --gpus-per-node=4
#SBATCH -t 0:10:00
#SBATCH --ntasks-per-node=1
#SBATCH -oReport-%j 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpu-bind=none

date
#module load anaconda3/2022.05 # Load module dependencies
module load pytorch
#conda activate diffusers 
conda env list
module list
which python
srun python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.__path__, torch.version.cuda)"
cat $0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 10000)) #12355

export OMP_NUM_THREADS=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
#export SLURM_CPU_BIND="cores"

srun python diffusion.py \
    --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
    --num_new_img_per_gpu 50 \
    --max_num_img_per_gpu 2 \
    --gradient_accumulation_steps 30 \
    #--resume outputs/model-N2000-device_count1-node8-epoch19-19004529 \
######################################################################################
