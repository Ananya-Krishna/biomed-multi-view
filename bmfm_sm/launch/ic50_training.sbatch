#!/bin/bash
#SBATCH --job-name=ic50_pipeline
#SBATCH --output=ic50_pipeline_%j.out
#SBATCH --error=ic50_pipeline_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ananya.krishna@yale.edu

module purge
module load miniconda/24.7.1

# use the GPFS env’s python directly
GPFS_PYTHON="/gpfs/gibbs/project/spiegel/ark89/biomed-multi-view/biomed-multi-view/envs/biomed-multiview/bin/python"

cd /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES

$GPFS_PYTHON ic50_training_v4.py \
  --emb-dir /gpfs/gibbs/project/spiegel/ark89/biomed-multi-view/biomed-multi-view/mmp_finetune_output \
  --chembl-dir /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES/chembl_built \
  --output-dir /gpfs/gibbs/project/spiegel/ark89/IBM/t-SMILES/ic50_pipeline_results_v5 \


