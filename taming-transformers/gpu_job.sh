#!/bin/bash   
#SBATCH -t 40:00:00                  # walltime = 6 hours
#SBATCH -N 1                         #  two node
#SBATCH -c 16    #  sixteen CPU (hyperthreaded) cores
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=kellislab
python main.py --base configs/multiscale_vqvae.yaml --resume logs/2025-05-12T14-59-40_multiscale_vqvae/checkpoints/last.ckpt -t True --gpus 0,1,2,3 --max_epochs 60 # Run the job steps 

