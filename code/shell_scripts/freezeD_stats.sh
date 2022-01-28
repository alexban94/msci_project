#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=8g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0


#echo "Beginning test for gestational face data"
#python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_face --config=configs/test.yml --results_dir=results/test_extensions --gpu 0

echo "Pre computing statistics for FID and KID for full augmentation"
python evaluations/calc_ref_stats.py --dataset gestational_face --n_classes 5 --inception_model_path datasets/inception_model --augmentation full
python evaluations/calc_ref_stats.py --dataset gestational_ear --n_classes 5 --inception_model_path datasets/inception_model --augmentation full
python evaluations/calc_ref_stats.py --dataset gestational_foot --n_classes 5 --inception_model_path datasets/inception_model --augmentation full

echo "Pre computing statistics for FID and KID for partial augmentation"
python evaluations/calc_ref_stats.py --dataset gestational_face --n_classes 5 --inception_model_path datasets/inception_model --augmentation partial
python evaluations/calc_ref_stats.py --dataset gestational_ear --n_classes 5 --inception_model_path datasets/inception_model --augmentation partial
python evaluations/calc_ref_stats.py --dataset gestational_foot --n_classes 5 --inception_model_path datasets/inception_model --augmentation partial

echo "Pre computing statistics for FID and KID for no augmentation"
python evaluations/calc_ref_stats.py --dataset gestational_face --n_classes 5 --inception_model_path datasets/inception_model --augmentation none
python evaluations/calc_ref_stats.py --dataset gestational_ear --n_classes 5 --inception_model_path datasets/inception_model --augmentation none
python evaluations/calc_ref_stats.py --dataset gestational_foot --n_classes 5 --inception_model_path datasets/inception_model --augmentation none


