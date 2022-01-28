#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=8g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0

# Need to calculate inception stats for FID/KID - get the mean/std/activations for both the overall data and class specific prior to this.
cd ..
python evaluate_trained_gans.py --results_dir=results/GANs_evaluation --gpu 0



