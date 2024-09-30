#!/bin/bash
#SBATCH -A m4717 
#SBATCH -J diffusion
#SBATCH -C gpu&hbm40g
#SBATCH -q regular #shared 
#SBATCH -N1
#SBATCH --gpus-per-node=4
#SBATCH -t 06:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -oReport-%j 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpu-bind=none

date
#module load anaconda3/2022.05 # Load module dependencies
module load pytorch #/2.0.1
#conda activate diffusers 
which python
conda env list
module list
#srun python -c "import torch; print('device_count', torch.cuda.device_count(), 'torch.__path__', torch.__path__, 'cuda version', torch.version.cuda); import accelerate; print('accelerate.__version', accelerate.__version__, 'accelerate.__path__', accelerate.__path__)"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 10000)) #12355
#export OMP_NUM_THREADS=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
#export SLURM_CPU_BIND="cores"
#echo $MASTER_ADDR
#echo $MASTER_PORT
#nc -zv $MASTER_ADDR $MASTER_PORT

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
cat $0
#nvidia-smi

srun python diffusion.py \
    --num_image 6400 \
    --batch_size 5 \
    --n_epoch 50 \
    --gradient_accumulation_steps 10 \
    --num_new_img_per_gpu 200 \
    --max_num_img_per_gpu 100 \
    --channel_mult 1 1 2 2 4 4 \
    --autocast 1 \
    --use_checkpoint 1 \
    --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
    #--resume outputs/model-N6400-device_count4-node2-epoch49-29150501 \

date
