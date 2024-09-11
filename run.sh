#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=16           # Number of CPU to request for the job
#SBATCH --mem=128GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=05-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --output=logs/test_%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --prefer=48gb
################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchlong                 #  researchlong pradeepresearch
#SBATCH --account=pradeepresearch   # The account you've been assigned (normally student)
#SBATCH --qos=mhhoang-20240209        # mhhoang-20240209 research-1-qos pradeepresearch-priority
#SBATCH --job-name=UNIQ     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Anaconda3/2022.05

# Create a virtual environment can be commented off if you already have a virtual environment
# conda create -n myenvnamehere

# Do not remove this line even if you have executed conda init
eval "$(conda shell.bash hook)"

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
conda activate IQ

# conda install -c conda-forge glew
# conda install -c conda-forge mesalib
# conda install -c menpo glfw3
# export CPATH=$CONDA_PREFIX/include
# pip install patchelf


srun whichgpu
srun --gres=gpu:1 python -u run_experiments.py

