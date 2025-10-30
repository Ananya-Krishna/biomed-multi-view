#!/bin/bash
#SBATCH --job-name=ic50_cls_fp
#SBATCH --output=ic50_cls_fp_%j.out
#SBATCH --error=ic50_cls_fp_%j.err
#SBATCH --partition=day          # use a CPU partition on your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ananya.krishna@yale.edu

module purge
module load miniconda/24.7.1

# Python with RDKit + PyTorch (CPU) installed
GPFS_PYTHON="/gpfs/gibbs/project/spiegel/ark89/biomed-multi-view/biomed-multi-view/envs/biomed-multiview/bin/python"

# Use all allocated CPU threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES

$GPFS_PYTHON ic50_training_fingerprints_classification.py \
  --chembl-dir /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES/chembl_built \
  --output-dir /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES/ic50_cls_results_v1 \
  --fps ecfp2048_r2,maccs \
  --split scaffold \
  --split-seed 42 \
  --seeds 3 \
  --threshold 6.0 \
  --cmp ge \
  --prob-threshold 0.5
