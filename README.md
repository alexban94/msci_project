# Investigating the use of generative adversarial networks to augment small datasets
Reupload of code used for my masters thesis: Investigating the use of generative adversarial networks to augment small datasets. Originally forked from: https://github.com/sangwoomo/FreezeD and credit to the authors of [Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs](https://arxiv.org/abs/2002.10964), Sangwoo Mo, Minsu Cho, Jinwoo Shin. See this original repository if you wish to use FreezeD with your own dataset; in order to run this code it is necessary to download their pretrained GAN and inceptionv3 models.

The project used the GesATional Dataset provided my supervisor Mercedes Torres Torres based on her previous work: [Postnatal gestational age estimation of newborns using Small Sample Deep Learning](https://www.sciencedirect.com/science/article/pii/S0262885618301483).

Multiple GAN models were trained using the FreezeD and TransferGAN techniques, which were then utilized to generate images to augment datasets when training ResNet-18 models to observe whether performance improved.

The changes made to the original software and new files are indicated and explained, followed by instructions on how to repeat the the experiments performed for my thesis.

The CUDA version used was `cuda-10.0`, with CUDNN version `cudnn-v7.6.5.32-forcuda10.0`. It is recommended that these are used.
The file `PYTHON_ENV.txt` shows all python libraries installed while running the software; refer to the versions used in this file if having issues.

## Main modifications to original software:
Note that the modifications were made specifically for use with the GestATional dataset. If not using this dataset, it is recommended that the original FreezeD repository be used instead.

1. The `datasets/make_image_list.py` file is run to generate a text file containing the image paths and labels of all files to be used in training the GANs. This was modified to:
   * Convert the folder names (classes) into integer labels.
   * Take an extra parameter to specify the degree of augmentation to use (i.e. the number of rotations to keep in the dataset), and prepare the data as such.
   * An additional file, `make_image_list_cv_split.py`, was created to use when preparing the image lists for training the GANs used in the classification experiments.       This partitions the dataset using stratified kfold sampling, and produces three image lists of training data for the dataset specified – one for each fold.
2. The `evaluation.py` file was extended to:
   * The original *get_mean_cov* function gets the mean and covariance of the inceptionv3 activations of the input dataset. This was changed to *get_mean_cov_acts*, to return the activations too, as it is necessary for KID calculation.
   * *calc_FID*  was changed to allow FID to be averaged over a number of evaluations, and allows it to be run when not used as a chainer extension.
   * *calc_KID* was added as a chainer extension, and operates similarly to *calc_FID* but instead computes KID. Note that the actual computation of KID was provided from code in the repository (metric and kid_score files) but had to be adapted for use with chainer.
3. The `evaluations/calc_ref_stats.py` file is used to precalculate FID/KID statistics on the training data used. It imports the *get_mean_cov_acts* function previously mentioned to do this.
   * Previously only FID statistics were saved. This was updated to save KID statistics for use during training.
   * Also takes augmentation strategy as a parameter for specifying the image list to load and directory to save in.
4. The `finetune_v2.py` file is used to train the GANs. The chainer extensions used were modified:
   * A chainer extension was added to calculate the KID every evaluation step.
   * Extra report keys added to accommodate extra results from the FID/KID calculated (mean/std/min).
   * The best model is saved according to the best KID instead of FID, and a snapshot is saved every evaluation step.
5. A script, `evaluate_trained_gans.py` was added. For an input directory of trained GAN outputs (after `finetune_v2.py` is run), calculates the KID/FID class-wise and overall for each experiment.
   * Takes the results directory as input as location to save .csv results.
   * The trained model directories must be placed in the models_to_eval folder, and the associated config file must be named `config.yml`.
6. A `train_resnet.py` file was added. This trains a resnet-18 model for each fold using stratified kfold sampling for the input dataset, and the results are aggregated to calculate the confusion matrix and accuracy/f-1score.
   * The *gen_image_with_condition* function from the FreezeD repo was adapted to generate a single image only and allow the use of the truncation trick. The *sample_continious* function in `source/miscs/random_samples.py` was also modified to this end.
   * Uses the associated generator to generate the required images during runtime, rather than loading from a directory of saved images.
7. The `resnet_experiments.py` file is an extension of `train_resnet.py`. 
   * The main change is to specify the experiments to run in a python dictionary, rather than via command line. This allows the results to be aggregated together, rather than individually. 
   * Each experiment is also run 5 times to calculate an average of the performance with standard deviation.

## How to run experiments
To repeat the experiments, all scripts are prepared in the shell_scripts folder. If not using slurm and instead running on your own machine, only run the python commands. Note that the config files are included for all these experiments and no changes need to be made.

1. The GestATional dataset must be placed in the datasets folder,  with the folder names gestational_face, gestational_ear, gestational_foot. The subdirectories must be the class names, extremely, very, moderate, term, late. 
2. To repeat the GAN experiments on the entire dataset, the `freezeD_full/none/partial.sh` scripts can be run. 
a. The image lists are prepared, followed by precalculating the KID/FID statistics, then the FreezeD/Finetuning experiments.
3. Move the output of the previous experiments to the models_to_eval folder and run evaluate_shell.sh to calculate average FID/KID of the best generator.
4. For training the GANs used in classification experiments,  first run `cv_split_GAN.sh` to prepare the image lists needed.
5. Next, run `train_cv_split_GANs.sh` to calculate KID/FID statistics and train all models (face/ear/foot) for each training data split (0/1/2).
6. To run the individual classification experiments (the train_resnet.py script), run `resnet_face/ear/foot/_experiments.sh`.
7. To run the classification experiments together with averaged aggregated output (`resnet_experiments.py`), then run `resnet_scratch.sh` or `resnet_pretrained.sh`.
8. To run classification experiments based on number of generated images used in augmentation, `run resnet_ims.sh`, and `resnet_ims_trunc.sh` to run with the truncation trick.

## Generated Examples

