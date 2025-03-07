#!/bin/bash
#SBATCH -J nanosphere_smooth_0_01ev_coarse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1G
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhan.tseng@yale.edu

module load miniconda
conda activate microsphere
python ../project_dm_smooth_ana.py
