#!/bin/bash
#SBATCH -J nanosphere_20250103_nll_coarse_0_01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python ../calc_profile_nlls.py sphere_20250103 0.01 coarse

