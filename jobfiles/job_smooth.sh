#!/bin/bash
#SBATCH -J nanosphere_generate_noise_1e9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python ../project_dm_smooth.py
