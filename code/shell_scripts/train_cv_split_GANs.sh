#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0


# FACE
#echo "Beginning face dataset"
#echo "Pre computing statistics for FID and KID"
#python evaluations/calc_ref_stats_cv.py --dataset gestational_face --n_classes 5 --inception_model_path datasets/inception_model --augmentation partial --split 0
#python evaluations/calc_ref_stats_cv.py --dataset gestational_face --n_classes 5 --inception_model_path datasets/inception_model --augmentation partial --split 1
#python evaluations/calc_ref_stats_cv.py --dataset gestational_face --n_classes 5 --inception_model_path datasets/inception_model --augmentation partial --split 2

#echo "Begin training face split 0"
#python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_face --config=configs/cv/split_0_face.yml --results_dir=results/split_0_face --gpu 0
#echo "Begin training face split 1"
#python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_face --config=configs/cv/split_1_face.yml --results_dir=results/split_1_face --gpu 0
#echo "Begin training face split 2"
#python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_face --config=configs/cv/split_2_face.yml --results_dir=results/split_2_face --gpu 0


# EAR
echo "Beginning ear dataset"
#echo "Pre computing statistics for FID and KID"
#python evaluations/calc_ref_stats_cv.py --dataset gestational_ear --n_classes 5 --inception_model_path datasets/inception_model --augmentation full --split 0
#python evaluations/calc_ref_stats_cv.py --dataset gestational_ear --n_classes 5 --inception_model_path datasets/inception_model --augmentation full --split 1
#python evaluations/calc_ref_stats_cv.py --dataset gestational_ear --n_classes 5 --inception_model_path datasets/inception_model --augmentation full --split 2

echo "Begin training ear split 0"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_ear --config=configs/cv/split_0_ear.yml --results_dir=results/split_0_ear --layer 3 --gpu 0 
echo "Begin training ear split 1"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_ear --config=configs/cv/split_1_ear.yml --results_dir=results/split_1_ear --layer 3 --gpu 0
echo "Begin training ear split 2"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_ear --config=configs/cv/split_2_ear.yml --results_dir=results/split_2_ear --layer 3 --gpu 0


# FOOT
echo "Beginning foot dataset"
echo "Pre computing statistics for FID and KID"
python evaluations/calc_ref_stats_cv.py --dataset gestational_foot --n_classes 5 --inception_model_path datasets/inception_model --augmentation full --split 0
python evaluations/calc_ref_stats_cv.py --dataset gestational_foot --n_classes 5 --inception_model_path datasets/inception_model --augmentation full --split 1
python evaluations/calc_ref_stats_cv.py --dataset gestational_foot --n_classes 5 --inception_model_path datasets/inception_model --augmentation full --split 2

echo "Begin training ear split 0"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_foot --config=configs/cv/split_0_foot.yml --results_dir=results/split_0_foot --layer 3 --gpu 0 
echo "Begin training ear split 1"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_foot --config=configs/cv/split_1_foot.yml --results_dir=results/split_1_foot --layer 3 --gpu 0
echo "Begin training ear split 2"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_foot --config=configs/cv/split_2_foot.yml --results_dir=results/split_2_foot --layer 3 --gpu 0

