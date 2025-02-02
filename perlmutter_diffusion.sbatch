#!/bin/bash
#SBATCH -A m4717 
#SBATCH -C gpu&hbm80g
#SBATCH -q regular #shared
#SBATCH --gpus-per-node=4
#SBATCH -N2
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -oReport-%j 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpu-bind=none

cat $0

module load pytorch #/2.0.1
which python
conda env list
module list

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 10000)) #12355
#export OMP_NUM_THREADS=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=DETAIL

git diff diffusion.py

srun python diffusion.py \
    --num_image 3200 \
    --num_new_img_per_gpu 32 \
    --max_num_img_per_gpu 32 \
    --model_channels 128 \
    --channel_mult 1 2 3 4 \
    --num_res_blocks 1 \
    --num_redshift 32 \
    --stride 2 2 2 \
    --guide_w 0 \
    --batch_size 32 \
    --n_epoch 120 \
    --gradient_accumulation_steps 1 \
    --lrate 1e-5 \
    --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
    --sample 1 \
    --ema 1 \
    #--resume ./outputs/model-N3200-device_count4-node2-epoch119-35015390 \
    #--autocast 1 \
    #--use_checkpoint 1 \
    #--dropout 0 \

sacct -j $SLURM_JOB_ID --format=JobID,JobName,Start,Elapsed
# SHORT NOTES:
#
