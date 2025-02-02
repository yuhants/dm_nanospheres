#!/bin/bash

#SBATCH --partition=day
#SBATCH -t 1-00:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --output=vscode_slurm.txt

# vscode_slurm.sh

# Usage:

# sbatch vscode_slurm.sh

# After this script successfully starts running, use the last line of the
#  the logfile 'vscode_slurm.txt' (in the directory you submitted the job from)
#  to set up a connection from the cluster to your own VScode app on a remote computer.
#  An example last line will look like:

######################
# vscode_slurm.txt
######################
# ...
# To grant access to the server, please log into https://github.com/login/device and use code ​XXXX-XXXX
######################

# Note: if you will use a shared partition for this job, we suggest 
#  the 'day' queue. If many users start requesting multi-day jobs with this method,
#  it becomes more likely some users may accidently leave their
#  VSCode jobs running when not in use.
# This could lead to unnecessary resource consumption on the computing clusters.

module load VSCode

code tunnel
