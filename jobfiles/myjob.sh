#!/bin/bash

#SBATCH --job-name=test_modulation
#SBATCH --time=0:10:00
#SBATCH --mem=2G
#SBATCH --mail-type=ALL

module load miniconda
conda activate microsphere
python ../dm_modulation_rate.py 1736027560.4007344