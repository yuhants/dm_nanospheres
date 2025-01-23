#!/bin/bash
#SBATCH -J nanosphere_alpha_nll
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python ../produce_limit.py 0.01
