import numpy as np
from PIL import Image
import chainer
import random
import scipy.misc


if __name__ == "__main__":
    import glob2, os, sys

    root_path = sys.argv[1]

    # Degree of augmentation: "none", "partial", "full"
    aug = sys.argv[2]

    # Modify this to use the absolute file path.
    abs_path = root_path
    print("Dataset: " + root_path)
    count = 0
    n_image_list = []
    filenames = glob2.glob(abs_path + '/**/*.png')
    
    if aug == "none":
        # Exclude rotations (which have "_r" in their name).
        print("No augmentation.")

        filenames = [name for name in filenames if not name.__contains__("_r")]
    elif aug=="partial":
        # Augment according to quantity per class to balance the data better.
        print("Partial augmentation")

        # Remove rotations and add back in the ones needed per class.
        filenames = [name for name in filenames if not name.__contains__("_r")]

        # Slicing to -4 removes the .PNG
        rotations = []
        for f in filenames:
            if f.__contains__("extremely"):
                # Add all rotations back in.
                for i in range(1, 10):
                    rotations.append(f[:-4] + "_r{}.PNG".format(i))
            elif f.__contains__("very"):
                # Add 4 rotations back in.
                rotations.append(f[:-4] + "_r1.PNG")
                rotations.append(f[:-4] + "_r3.PNG")
                rotations.append(f[:-4] + "_r7.PNG")
                rotations.append(f[:-4] + "_r9.PNG")

            else:
                # moderately/term/late, add 2 rotations.
                rotations.append(f[:-4] + "_r1.PNG")
                rotations.append(f[:-4] + "_r9.PNG")
        # Add rotations back to filenames.
        filenames = filenames + rotations

    elif aug=="full":
        # Use all rotations; do nothing
        print("Full augmentation.")
    else:
        exit("ERROR: dataset augmentation not specified")

    # Count class distribution:
    print("Images in each class:")
    for label in ["extremely", "very", "moderate", "term", "late"]:
        count = 0
        for f in filenames:
            if label in f:
                count += 1
        print(label + ": {}".format(count))

    # Conversion dictionary for string label name into int.
    # extremely - very - moderate - term - very
    labels = {
        "extremely": 0,
        "late": 4,
        "moderate": 2,
        "term": 3,
        "very": 1
    }

    count = 0
    for filename in filenames:
        filename = filename.split('/')
        if "gest" in root_path:
            # Convert from class name to int representation.
            dirname = filename[-2]
            label = labels[dirname]
        else:
            dirname = filename[-2]
            label = dirname  
        n_image_list.append([os.path.join(filename[-2], filename[-1]), label])
        count += 1

    print("Num of examples:{}".format(count))
    print("----------------------\n")
    n_image_list = np.array(n_image_list, np.str)
    np.savetxt(f'image_list_{root_path}_{aug}.txt', n_image_list, fmt="%s")

