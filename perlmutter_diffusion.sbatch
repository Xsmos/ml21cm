#!/bin/bash
#SBATCH -A m4717 
#SBATCH -J diffusion
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N2 
#SBATCH --gpus-per-node=2
#SBATCH -t 0:05:00
#SBATCH --ntasks-per-node=1
#SBATCH -oReport-%j 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpu-bind=none

date
#module load anaconda3/2022.05 # Load module dependencies
module load pytorch #/2.0.1
#conda activate diffusers 
conda env list
module list
which python
srun python -c "import torch; print('device_count', torch.cuda.device_count(), 'torch.__path__', torch.__path__, 'cuda version', torch.version.cuda); import accelerate; print('accelerate.__version', accelerate.__version__, 'accelerate.__path__', accelerate.__path__)"
cat $0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 10000)) #12355
#export OMP_NUM_THREADS=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
#export SLURM_CPU_BIND="cores"
echo $MASTER_ADDR
echo $MASTER_PORT
#nc -zv $MASTER_ADDR $MASTER_PORT

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

srun python diffusion.py \
    --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
    --num_new_img_per_gpu 50 \
    --max_num_img_per_gpu 2 \
    --gradient_accumulation_steps 30 \
    #--resume outputs/model-N2000-device_count1-node8-epoch19-19004529 \
    
