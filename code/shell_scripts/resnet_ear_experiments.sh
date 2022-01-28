#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0


echo "Begin resnet commands."
cd ..
echo "Pretrained experiments."
echo "No augmentation."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --output_name 'ear_no_aug' --pretrained True

echo "Rotations only."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --rotations True --output_name 'ear_rotate' --pretrained True

echo "Generated images only, n = 15."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 15 --augment_gen True --output_name 'ear_gen15' --pretrained True

echo "Generated images only, n = 30."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 30 --augment_gen True --output_name 'ear_gen30' --pretrained True

echo "Generated images only, n = 45."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 45 --augment_gen True --output_name 'ear_gen40' --pretrained True

echo " Rotations and Generated images, n = 30."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 30 --augment_gen True --rotations True --output_name 'ear_rot_gen' --pretrained True

#######################################
echo "From scratch."
echo "No augmentation."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --output_name 'scratch_ear_no_aug' --pretrained False

echo "Rotations only."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --rotations True --output_name 'scratch_ear_rotate' --pretrained False

echo "Generated images only, n = 15."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 15 --augment_gen True --output_name 'scratch_ear_gen15' --pretrained False

echo "Generated images only, n = 30."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 30 --augment_gen True --output_name 'scratch_ear_gen30' --pretrained False

echo "Generated images only, n = 45."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 45 --augment_gen True --output_name 'scratch_ear_gen40' --pretrained False

echo " Rotations and Generated images, n = 30."
python ./train_resnet.py --dataset gestational_ear --generated_dir ./models_to_eval --num_gen_images 30 --augment_gen True --rotations True --output_name 'scratch_ear_rot_gen' --pretrained False
