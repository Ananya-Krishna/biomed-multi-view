#!/bin/bash
#SBATCH --chdir=/gpfs/gibbs/project/spiegel/ark89/biomed-multi-view/biomed-multi-view
#SBATCH --job-name=finetune_mt
#SBATCH --output=logs/finetune_%j.out    # STDOUT → logs/finetune_<JOBID>.out
#SBATCH --error=logs/finetune_%j.err     # STDERR → logs/finetune_<JOBID>.err
#SBATCH --partition=bigmem                   # or gpu_devel / standard_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00 

# Load the miniconda module provided on Grace
module load miniconda/24.7.1

# Initialize conda (if needed) and activate your env
eval "$(conda shell.bash hook)"
conda activate /home/ark89/biomed-multi-view/envs/biomed-multiview

export NUMBA_DISABLE_JIT=1

# Ensure logs directory exists
mkdir -p logs

# Run the per-target finetuning+visualization
python bmfm_sm/launch/finetune_by_target.py \
  --pos-csv notebooks/mmp_ac_s_distinct.csv \
  --neg-csv notebooks/mmp_ac_s_neg_distinct.csv \
  --output-dir mmp_finetune_output \
  --embed-device cpu \
  --epochs 20 \
  --bs 32
