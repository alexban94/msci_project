#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0

### Experiments run only on the original data. ###############################################

### Preprocess datasets
### Data path: /home/alex/carey/home/psyajhi/data/gestational_...
#cd datasets
python make_image_list.py gestational_face "none"
python make_image_list.py gestational_ear "none"
python make_image_list.py gestational_foot "none"
#cd ..


### Pre-compute FID statistics and activations for KID.

#python source/inception/download.py --outfile=datasets/inception_model
echo "Pre computing statistics for FID and KID"
python evaluations/calc_ref_stats.py --dataset gestational_face --n_classes 5 --inception_model_path datasets/inception_model --augmentation none
python evaluations/calc_ref_stats.py --dataset gestational_ear --n_classes 5 --inception_model_path datasets/inception_model --augmentation none
python evaluations/calc_ref_stats.py --dataset gestational_foot --n_classes 5 --inception_model_path datasets/inception_model --augmentation none

### Run experiments
## Finetuning
# Face
echo "Beginning finetuning experiment for gestational face data"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_face --config=configs/none/sn_projection_gestational_face.yml --results_dir=results/none_gest_face_finetune --gpu 0

# Ear
echo "Beginning finetuning experiment for gestational ear data"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_ear --config=configs/none/sn_projection_gestational_ear.yml --results_dir=results/none_gest_ear_finetune --gpu 0

# Foot
echo "Beginning finetuning experiment for gestational foot data"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_foot --config=configs/none/sn_projection_gestational_foot.yml --results_dir=results/none_gest_foot_finetune --gpu 0

## FreezeD method
# Face
echo "Beginning freezeD experiment for gestational face data"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_face --config=configs/none/sn_projection_gestational_face.yml --results_dir=results/none_gest_face_freeze --layer 3 --gpu 0

# Ear
echo "Beginning freezeD experiment for gestational ear data"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_ear --config=configs/none/sn_projection_gestational_ear.yml --results_dir=results/none_gest_ear_freeze --layer 3 --gpu 0

# Foot
echo "Beginning freezeD experiment for gestational foot data"
python finetune_v2.py --data_dir=/cs/home/psyajhi/data/gestational_foot --config=configs/none/sn_projection_gestational_foot.yml --results_dir=results/none_gest_foot_freeze --layer 3 --gpu 0

