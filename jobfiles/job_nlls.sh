#!/bin/bash
#SBATCH -J nanosphere_20241202_nll_coarse_extended_alpha_right_10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python ../calc_profile_nlls.py sphere_20241202 10 coarse_extended_alpha_right

