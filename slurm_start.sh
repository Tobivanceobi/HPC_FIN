#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000
#SBATCH --tmp=20000
#SBATCH --time=05:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_%a.txt
#SBATCH --error err/error_%a.txt

# Remove previous results
#rm err/*; rm out/*; rm -r runs/*;

source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch

python3 $HOME/tobias_ettling/HPC_FIN/slurm_start.py $SLURM_ARRAY_TASK_ID $SLURM_NNODES

conda deactivate