#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0


### Preprocess datasets
### Data path: /home/alex/carey/home/psyajhi/data/gestational_...
cd datasets
python make_image_list_cv_split.py gestational_face "none"
python make_image_list_cv_split.py gestational_ear "none"
python make_image_list_cv_split.py gestational_foot "none"

python make_image_list_cv_split.py gestational_face "partial"
python make_image_list_cv_split.py gestational_ear "partial"
python make_image_list_cv_split.py gestational_foot "partial"

python make_image_list_cv_split.py gestational_face "full"
python make_image_list_cv_split.py gestational_ear "full"
python make_image_list_cv_split.py gestational_foot "full"
cd ..

