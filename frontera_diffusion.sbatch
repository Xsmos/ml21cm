#!/bin/bash
#SBATCH -J diffusion # Job name
#SBATCH -p rtx #-dev
#SBATCH -N4            # Number of nodes and cores per node required
#SBATCH --ntasks-per-node=1
#SBATCH -t 60:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -o logs/%x-%j
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=xiabin@gatech.edu

date
cat $0
mkdir -p logs
echo --------------------------------------------------------------------
 
#module load pytorch #/2.0.1
#conda activate diffusers
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
echo --------------------------------------------------------------------

srun --ntasks-per-node=1 torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    diffusion.py \
        --num_image 1600 \
        --num_new_img_per_gpu 8 \
        --max_num_img_per_gpu 2 \
        --channel_mult 1 2 3 4 \
        --num_redshift 64 \
        --stride 2 2 2 \
        --batch_size 2 \
        --n_epoch 3 \
        --gradient_accumulation_steps 16 \
        --lrate 1e-5 \
        --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
        #--resume ./outputs/model-N1600-device_count4-node4-36401694-epoch200.pt \
        #--guide_w 0 \
        #--num_res_blocks 1 \
        #--model_channels 128 \
        #--sample 1 \
        #--ema 0 \
        #--autocast 1 \
        #--use_checkpoint 1 \
        #--dropout 0 \

sacct -j $SLURM_JOB_ID --format=JobID,JobName,Start,End,Elapsed
