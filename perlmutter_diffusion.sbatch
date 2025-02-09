#!/bin/bash
#SBATCH -A m4717 
#SBATCH -C gpu&hbm80g
#SBATCH -q regular #debug #shared #
#SBATCH -N2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -t 3:00:00
#SBATCH -oReport-%j 
#SBATCH --mail-type=BEGIN,END,FAIL

cat $0
echo "$(printf '%0.s🚥 ' {1..30})"

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
#export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

git diff diffusion.py | cat

echo SLURM_NNODES=$SLURM_NNODES
echo SLURM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
echo SLURM_GPUS=$SLURM_GPUS
echo MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT

#srun python diffusion.py \
#srun --nodes=$SLURM_NNODES --gpus-per-node=$SLURM_GPUS_PER_NODE --ntasks-per-node=1 torchrun --nproc_per_node=1 \
#srun torchrun \
#torchrun \
#
#srun --gpus-per-node=4 --ntasks-per-node=1 --gpu-bind=none torchrun \
#    --nnodes=$SLURM_NNODES \
#    --nproc_per_node=1 \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
srun --gpus-per-node=4 --ntasks-per-node=1 --gpu-bind=none python \
    diffusion.py \
        --num_image 200 \
        --num_new_img_per_gpu 25 \
        --max_num_img_per_gpu 25 \
        --model_channels 128 \
        --channel_mult 1 2 3 4 \
        --num_res_blocks 1 \
        --num_redshift 64 \
        --stride 2 2 \
        --guide_w 0 \
        --batch_size 25 \
        --n_epoch 6 \
        --gradient_accumulation_steps 1 \
        --lrate 1e-5 \
        --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
        #--sample 1 \
        #--ema 0 \
        #--resume ./outputs/model-N3200-device_count4-node2-epoch119-35015390 \
        #--autocast 1 \
        #--use_checkpoint 1 \
        #--dropout 0 \

sacct -j $SLURM_JOB_ID --format=JobID,JobName,Start,End,Elapsed
