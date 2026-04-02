#!/bin/bash   
#SBATCH -t 10:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=kellislab
python demo_sample.py  
