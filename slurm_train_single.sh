#!/bin/bash
#SBATCH --job-name=FIN-Sing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=15:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_sing.txt
#SBATCH --error err/error_sing.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;

source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/HPC_FIN/train_single.py

conda deactivate