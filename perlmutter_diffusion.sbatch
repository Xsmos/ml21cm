#!/bin/bash
#SBATCH -J diffusion
#SBATCH -C gpu&hbm80g
#SBATCH -q regular #debug #shared #
#SBATCH -N10
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -t 2-0:00:00
#SBATCH -o training/logs/%x-%j
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p training/logs
date
cat $0
echo "$(printf '%0.s🚥 ' {1..30})"

module load pytorch #/2.3.1
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
export PYTHONPATH=$SCRATCH/ml21cm:$PYTHONPATH

git diff training/diffusion.py | cat

echo SLURM_JOB_NAME=$SLURM_JOB_NAME
echo SLURM_NNODES=$SLURM_NNODES
echo SLURM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
echo SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE
echo MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT
echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo SLURM_TRES_PER_TASK=$SLURM_TRES_PER_TASK

echo "$(printf '%0.s🚥 ' {1..30})"
srun --gpus-per-node=$SLURM_GPUS_ON_NODE --ntasks-per-node=1 --gpu-bind=none torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    training/diffusion.py \
        --num_image 640 \
        --batch_size 2 \
        --num_new_img_per_gpu 4 \
        --max_num_img_per_gpu 2 \
        --num_redshift 1024 \
        --channel_mult 1 1 2 4 \
        --n_epoch 120 \
        --stride 2 2 4 \
        --num_res_blocks 3 \
        --gradient_accumulation_steps 16 \
        --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
        --squish .1 0 \
        --scale_path "./utils/PowerTransformer_25600_z1.pkl" \
        --z_step 1 \
        --reset-epoch \
        --model_channels 128 \
        #--resume ./training/outputs/model-N800-device_count4-node8-40010823-epoch120.pt \
        #--sample 1 \
        #--autocast 1 \
        #--use_checkpoint 1 \
        #--dropout 0 \

date
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Start,End,Elapsed
