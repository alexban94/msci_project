import numpy as np
from PIL import Image
import chainer
import random
import scipy.misc
from sklearn.model_selection import StratifiedKFold



if __name__ == "__main__":
    import glob2, os, sys

    # The dataset to use (gestational_face/ear/foot)
    dataset = sys.argv[1]

    # Degree of augmentation: "none", "partial", "full"
    aug = sys.argv[2]

    # Modify this to use the absolute file path.
    abs_path = dataset
    print("Dataset: " + dataset)
    count = 0
    filenames = glob2.glob(abs_path + '/**/*.png')

    # Remove all rotations initially.
    filenames = [name for name in filenames if not name.__contains__("_r")]
    filenames = np.array(filenames)

    # Prepare the labels
    # Conversion dictionary for string label name into int.
    # extremely - very - moderate - term - late
    classes = {
        "extremely": 0,
        "late": 4,
        "moderate": 2,
        "term": 3,
        "very": 1
    }
    labels = []
    for filename in filenames:
        filename = filename.split('/')
        if "gest" in dataset:
            # Convert from class name to int representation.
            dirname = filename[-2]
            labels.append(classes[dirname])
    

    # Use stratified sampling to get k training/test folds 
    # Create the kfold object
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    split_index = 0
    for train_indices, test_indices in kfold.split(filenames, labels):

        # The test indices are not required here - only for testing the resnet.
        train_data = list(filenames[train_indices])
        train_labels = list(np.array(labels)[train_indices])
        
        if aug == "none":
            # Exclude rotations (which have "_r" in their name).
            print("No augmentation.")

        elif aug=="partial":
            # Augment according to quantity per class to balance the data better.
            print("Partial augmentation")

            # Slicing to -4 removes the .PNG
            rotations = []
            rotations_labels = []
            for f in train_data:
                if f.__contains__("extremely"):
                    # Add all rotations back in.
                    for i in range(1, 10):
                        rotations.append(f[:-4] + "_r{}.PNG".format(i))
                        rotations_labels.append(classes["extremely"])
                elif f.__contains__("very"):
                    # Add 4 rotations back in.
                    rotations.append(f[:-4] + "_r1.PNG")
                    rotations.append(f[:-4] + "_r3.PNG")
                    rotations.append(f[:-4] + "_r7.PNG")
                    rotations.append(f[:-4] + "_r9.PNG")
                    rotations_labels += 4 * [classes["very"]]

                else:
                    # moderate/term/late, add 2 rotations.
                    rotations.append(f[:-4] + "_r1.PNG")
                    rotations.append(f[:-4] + "_r9.PNG")
                    if f.__contains__("moderate"):
                        rotations_labels += 2 * [classes["moderate"]]
                    elif f.__contains__("term"):
                        rotations_labels += 2 * [classes["term"]]
                    elif f.__contains__("late"):
                        rotations_labels += 2 * [classes["late"]]
                    else:
                        exit("ERROR: data/label mismatch")                        
            # Add rotations and their labels to training data.
            train_data = train_data + rotations
            train_labels = train_labels + rotations_labels

        elif aug=="full":
            # Use all rotations; do nothing
            print("Full augmentation.")
            rotations = []
            rotations_labels = []
            for f in train_data:
                ro_files = glob2.glob(f[:-4] + '_r*')
                dirname = f.split('/')[-2]
                rotations = rotations + ro_files
                rotations_labels += len(ro_files) * [classes[dirname]]
            # Add rotations and their labels to training data.
            train_data = train_data + rotations # As train_data is a np array
            train_labels = train_labels + rotations_labels  

        else:
            exit("ERROR: dataset augmentation not specified")

        # Count class distribution:
        print("Images in each class:")
        for label in ["extremely", "very", "moderate", "term", "late"]:
            count = 0
            for f in train_data:
                if label in f:
                    count += 1
            print(label + ": {}".format(count))


        count = 0
        n_image_list = []

        for f in train_data:
            filename = f.split('/')
            if "gest" in dataset:
                # Convert from class name to int representation.
                dirname = filename[-2]
                label = classes[dirname]
            else:
                dirname = filename[-2]
                label = dirname  
            n_image_list.append([os.path.join(filename[-2], filename[-1]), label])
            count += 1

        print("Num of examples:{}".format(count))
        print("----------------------\n")
        n_image_list = np.array(n_image_list, np.str)
        np.savetxt(f'split_{split_index}_image_list_{dataset}_{aug}.txt', n_image_list, fmt="%s")
        split_index += 1

