#!/bin/bash   
#SBATCH -t 40:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:4
torchrun --nproc_per_node=4 train.py --auto_resume=False --ep 500 --fuse False 
