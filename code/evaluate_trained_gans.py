import os, sys, time
import shutil
import yaml
import random
import numpy as np
import cupy
from copy import deepcopy

import argparse
import chainer
from chainer import training, serializers
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from metric import compute_kid
from evaluation import *
from finetune_v2 import create_result_dir
import source.yaml_utils as yaml_utils
import glob2
import pandas as pd



# Script to calculate FID and KID from a trained model.
if __name__ == '__main__':

    # Fix randomness
    random.seed(0)
    np.random.seed(0)
    cupy.random.seed(0)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')
    parser.add_argument('--results_dir', type=str, default='./results/GAN_evaluation', help='directory to save the results to')
    # Load config, prepare result directory.
    args = parser.parse_args()
    out = args.results_dir
    #create_result_dir(out, args.config_path, config)

    # Lists for all results.
    all_kid_mean = []
    all_kid_std = []
    all_fid_mean = []
    all_fid_std = []
    # Retrieve the models.
    model_directories = glob2.glob('./models_to_eval/*/')
    print(model_directories)
    exp_names = []

    for path in model_directories:
        # Trained resnet generator path
        npz_file = path + 'ResNetGenerator_best.npz'
        config_path = path + 'config.yml'
        config = yaml_utils.Config(yaml.load(open(config_path), Loader=yaml.FullLoader))
        experiment_name = path.split('/')[-2]
        exp_names.append(experiment_name)
        print("Evaluating generator from experiment %s" % experiment_name, flush=True)

        # Get the stats directory, removing the specific file referenced by using [-8]
        # (easier than modifying all current config files).
        # Note: mean/cov per class: 0.npz, ..., 4.npz; Full mean/cov: full.npz
        #       activations per class: acts_0.npy, ..., acts_4.npy; Full acts: full_activations.npy
        stats_dir = config['eval']['stat_file'][:-8]  
        print(stats_dir)

        # Prepare config for loading the generator.
        gen_conf = deepcopy(config.models['generator'])
        gen_conf['args']['n_classes'] = 5
        gen_conf['args'].pop('n_classes_src')
        gen_conf['args'].pop('n_classes_tgt')

        chainer.cuda.get_device_from_id(args.gpu).use()

        # Instantiate the same base network as what was trained.
        gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
        gen.to_gpu(device=args.gpu)

        # Then load the saved best parameters into the instance.
        serializers.load_npz(npz_file, gen)

        # Save a sample of conditional generation for this generator.
        sample_generate_conditional(gen, out, filename=experiment_name)

        # Load inception model for obtaining stats on generated images.
        model = load_inception_model('datasets/inception_model')

        # Number of times to compute KID/FID for average, number of images to use.
        num_evals = 5
        num_images = 5000

        # Get KID/FID stats per class
        class_kid_mean = []
        class_kid_std = []
        class_fid_mean = []
        class_fid_std = []

        for c in range(5): 
            # Load stat files for class c.
            fid_stat = np.load(stats_dir + '%i.npz' % c)
            kid_stat = np.load(stats_dir + 'acts_%i.npy' % c)
            
            fid_results = []
            kid_results = []

            for i in range(1): # should be num_evals, but using 1 temporarily.
                print("On evaluation %i for class %i" % (i, c))
                ims = gen_images_with_condition(gen, c, num_images).astype("f")
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    mean, cov, acts_fake = get_mean_cov_acts(model, ims)

                # Calculate FID and store result.
                fid = FID(fid_stat["mean"], fid_stat["cov"], mean, cov)
                fid_results.append(fid)

                # Calculate KID
                # Perform KID here using activations.
                kid = compute_kid(kid_stat, acts_fake)
                kid_results.append(kid)
            
            # Store class specific results.
            class_fid_mean.append(np.mean(fid_results))
            class_fid_std.append(np.std(fid_results))

            class_kid_mean.append(np.mean(kid_results))
            class_kid_std.append(np.std(kid_results))


        # Get KID/FID stats overall.
        fid_stat = np.load(stats_dir + 'full.npz')
        kid_stat = np.load(stats_dir + 'full_activations.npy')


        print("Beginning %i overall FID/KID evaluations." % num_evals)
        fid_results = []
        kid_results = []

        num_evals = 5
        num_images = 5000
       
        for i in range(num_evals):
            print("On evaluation %i" % i)
            ims = gen_images(gen, num_images).astype("f")
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                mean, cov, acts_fake = get_mean_cov_acts(model, ims)

            # Calculate FID and store result.
            fid = FID(fid_stat["mean"], fid_stat["cov"], mean, cov)
            fid_results.append(fid)

            # Calculate KID
            # Perform KID here using activations.
            kid = compute_kid(kid_stat, acts_fake)
            kid_results.append(kid)
        
        fid_mean = np.mean(fid_results)
        fid_std = np.std(fid_results)

        kid_mean = np.mean(kid_results)
        kid_std = np.std(kid_results)


        # Aggregate class specific and overall results.
        print("For experiment %s, evaluation results:" % experiment_name)
        all_fid_mean.append(class_fid_mean + [fid_mean])
        
        all_fid_std.append(class_fid_std + [fid_std])

        all_kid_mean.append(class_kid_mean + [kid_mean])

        all_kid_std.append(class_kid_std + [kid_std])
      
        # Index -1 to get this iterations results (most recently added to the list)
        print("All FID means")
        print(all_fid_mean[-1])
        print("All FID stds")
        print(all_fid_std[-1])
        print("All KID means")
        print(all_kid_mean[-1])
        print("All KID stds")
        print(all_kid_std[-1])
    
    # Currently all results are in lists of lists
    # Store all results in dataframes and save as csv.
    print("Finished evaluations - saving results.")
    cols = ["0", "1", "2", "3", "4", "Overall"]
    
    print(all_fid_mean)
    df_fid_mean = pd.DataFrame(all_fid_mean, columns=cols, index=exp_names)
    df_fid_mean.to_csv(out + "/fid_means.csv")

    df_fid_std = pd.DataFrame(all_fid_std, columns=cols, index=exp_names)
    df_fid_std.to_csv(out + "/fid_stds.csv")

    df_kid_mean = pd.DataFrame(all_kid_mean, columns=cols, index=exp_names)
    df_kid_mean.to_csv(out + "/kid_means.csv")

    df_kid_std = pd.DataFrame(all_kid_std, columns=cols, index=exp_names)
    df_kid_std.to_csv(out + "/kid_stds.csv")

    








