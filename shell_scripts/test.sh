#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0


echo "test"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_face --config=configs/cv/split_0_face.yml --results_dir=results/split_0_face --gpu 0

