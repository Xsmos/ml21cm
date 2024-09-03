#!/bin/bash
#SBATCH -J diffusion # Job name
#SBATCH -A gts-jw254-coda20 
#SBATCH -qembers
#SBATCH -N2 --gpus-per-node=V100:2 -C V100-16GB              # Number of nodes and cores per node required
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=16G                        # Memory per core
#SBATCH -t 00:10:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences

#module load gcc/10.3.0-o57x6h
#module load mvapich2/2.3.6-ouywal

pwd
date
module load anaconda3/2023.03 #anaconda3/2022.05 # Load module dependencies
#module load pytorch

conda activate diffusion 
conda env list
module list
srun python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.__path__, torch.version.cuda)"
cat $0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 10000)) #12355

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

srun python diffusion.py \
    --train "$SCRATCH/LEN128-DIM64-CUB16-Tvir[4, 6]-zeta[10, 250]-0809-123640.h5" \
    --num_new_img_per_gpu 50 \
    --max_num_img_per_gpu 2 \
    --gradient_accumulation_steps 30 \
    #--resume outputs/model-N2000-device_count1-node8-epoch19-19004529 \

