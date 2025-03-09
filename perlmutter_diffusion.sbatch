#!/bin/bash
#SBATCH -A m4717 
#SBATCH -J diffusion
#SBATCH -C gpu #&hbm80g
#SBATCH -q regular #debug #shared #
#SBATCH -N4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j
#SBATCH --mail-type=BEGIN,END,FAIL

date
cat $0
mkdir -p logs
echo "$(printf '%0.s🚥 ' {1..30})"

module load pytorch #/2.0.1
which python
which torchrun
conda env list
module list

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 10000)) #12355 #29500 
#export OMP_NUM_THREADS=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

git diff diffusion.py | cat

echo SLURM_JOB_NAME=$SLURM_JOB_NAME
echo SLURM_NNODES=$SLURM_NNODES
echo SLURM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
echo SLURM_GPUS=$SLURM_GPUS
echo MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT
#env | grep SLURM_
echo "$(printf '%0.s🚥 ' {1..30})"

srun --gpus-per-node=4 --ntasks-per-node=1 --gpu-bind=none torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    diffusion.py \
        --num_image 1600 \
        --batch_size 2 \
        --num_new_img_per_gpu 8 \
        --max_num_img_per_gpu 8 \
        --lrate 1e-5 \
        --num_redshift 1024 \
        --channel_mult 1 2 3 4 \
        --n_epoch 120 \
        --stride 2 2 \
        --num_res_blocks 3 \
        --ema 1 \
        --gradient_accumulation_steps 32 \
        --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
        #--resume ./outputs/model-N1600-device_count4-node4-36674041-epoch120.pt \
        #--guide_w 0 \
        #--model_channels 128 \
        #--sample 1 \
        #--autocast 1 \
        #--use_checkpoint 1 \
        #--dropout 0 \

sacct -j $SLURM_JOB_ID --format=JobID,JobName,Start,End,Elapsed
