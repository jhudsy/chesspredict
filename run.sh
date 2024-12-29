#!/bin/bash
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=2 # number of cores
#SBATCH --mem=64G # memory pool for all cores

#SBATCH --ntasks-per-node=1 # one job per node
#SBATCH --gres=gpu:2 
#SBATCH --partition=a100_mig

#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.oren@abdn.ac.uk

module load python3
source ~/chesspredict/.venv/bin/activate

srun python main.py 
