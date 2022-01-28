#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0


echo "Begin resnet commands."
cd ..


python ./resnet_experiments_ims.py --generated_dir ./models_to_eval --output_name 'resnet_experiments_trunc_pretrained' --pretrained True --truncate True
python ./resnet_experiments_ims.py --generated_dir ./models_to_eval --output_name 'resnet_experiments_trunc_scratch' --pretrained False --truncate True
