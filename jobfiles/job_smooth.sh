#!/bin/bash
#SBATCH -J nanosphere_smooth_0_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python ../project_dm_smooth.py 0.1
