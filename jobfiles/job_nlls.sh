#!/bin/bash
#SBATCH -J nanosphere_20250103_nll_coarse_extended_alpha_right_10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python ../calc_profile_nlls_dm_only.py sphere_20250103 10 coarse_extended_alpha_right

