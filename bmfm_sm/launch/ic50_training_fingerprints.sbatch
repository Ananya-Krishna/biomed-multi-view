#!/bin/bash
#SBATCH --job-name=ic50_fp
#SBATCH --output=ic50_fp_%j.out
#SBATCH --error=ic50_fp_%j.err
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ananya.krishna@yale.edu

module purge
module load miniconda/24.7.1

# If this env already has RDKit + PyTorch, use it. Otherwise swap to an env that does.
GPFS_PYTHON="/gpfs/gibbs/project/spiegel/ark89/biomed-multi-view/biomed-multi-view/envs/biomed-multiview/bin/python"

# (Optional but helpful on shared nodes)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES

$GPFS_PYTHON ic50_training_fingerprints.py \
  --chembl-dir /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES/chembl_built \
  --output-dir /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES/ic50_fp_results_v2 \
  --fps ecfp2048_r2,maccs \
  --split scaffold \
  --split-seed 42 \
  --seeds 3
