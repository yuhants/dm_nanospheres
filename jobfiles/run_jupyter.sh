#!/bin/bash
#SBATCH --partition day
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 6:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$CLUSTER
token=$(echo $RANDOM | md5sum | head -c 30)

# print tunneling instructions jupyter-log
echo -e "
For more info and how to connect from windows,
   see https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.ycrc.yale.edu
Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.ycrc.yale.edu
SSH login: $user
SSH port: 22
Use a Browser on your local machine to go to:
localhost:${port}/lab?token=${token}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
module load miniconda; conda activate microspheres


export JUPYTER_TOKEN=$token
jupyter lab --no-browser --port=${port} --ip=${node}
