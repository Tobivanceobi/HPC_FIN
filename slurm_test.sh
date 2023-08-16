#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=00:10:00
#SBATCH --array=0-2
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_%a.txt
#SBATCH --error err/error_%a.txt

# Remove previous results
#rm err/*; rm out/*; rm -r runs/*;

source $HOME/tobias_ettling/HPC-FIN/venv/bin/activate
python3 $HOME/tobias_ettling/HPC-FIN/slurm_test.py $SLURM_ARRAY_TASK_ID $SLURM_NNODES
deactivate
