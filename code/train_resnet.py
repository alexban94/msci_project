import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import glob2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from skimage import io
import argparse
import time
import copy
import os
from PIL import Image

# Required for the generator
from copy import deepcopy
import chainer
from chainer import training, serializers

import yaml
import source.yaml_utils as yaml_utils
import glob2

from source.miscs.random_samples import sample_continuous


# Based on example: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

# Need to define custom class for the dataloader.
class GestationalDataset(Dataset):
    def __init__(self, filenames, labels, transform=None, gen_data=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.gen_data_dict = gen_data
     

    def __len__(self):
        return len(self.filenames) # Number of images.
   
    def __getitem__(self, index):
        img_path = self.filenames[index]
        #print(img_path)
        # Generated images are added to the dataset by the string 'generate_image' with the appropriate label.
        if "generated_image" in img_path:
            label = torch.tensor(self.labels[index])
            # Use the img_path identifier to get the associated image from the dictionary.
            image = self.gen_data_dict[img_path]
        else:
            # img_path is the file location; load it.
            image = Image.open(img_path)
            label = torch.tensor(self.labels[index])

        # If there is a transform apply it to the loaded image.
        if self.transform:
            image = self.transform(image)

        return (image, label)


# Note: this is adapted from the function in evaluation.py from the FreezeD projection repository.
def gen_image_with_condition(gen, c=0, n=1, batchsize=1, trunc_flag=False, threshold=0.3):
    # Use numpy/cupy
    xp = gen.xp
    # Disable training config and do a forward pass to generate an image.
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = xp.asarray([c], dtype=xp.int32)
        # Normally, the image is sampled with no truncation threshold if z is not specified.
        # Create own noise vector z.
        z = sample_continuous(128, 1, distribution="normal", xp=xp, trunc_flag=trunc_flag, threshold=threshold) # z_dim, batchsize, distribution type (normal/uniform), np/cupy
        x = gen(batchsize, y=y, z=z)
    x = chainer.cuda.to_cpu(x.data)
    # Convert to range 0 to 255.
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)[0]
    # Reorder the axes so that it can be used by PIL.
    x = np.moveaxis(x, 0, -1)
    
    return x

def load_generator(gen_folder):
    # Import required chainer libraries and load the generator.
    npz_file = gen_folder + '/ResNetGenerator_best.npz'
    config_path = gen_folder + '/config.yml'
    config = yaml_utils.Config(yaml.load(open(config_path), Loader=yaml.FullLoader))

    # Prepare config for loading the generator.
    gen_conf = deepcopy(config.models['generator'])
    gen_conf['args']['n_classes'] = 5
    gen_conf['args'].pop('n_classes_src')
    gen_conf['args'].pop('n_classes_tgt')

    chainer.cuda.get_device_from_id(0).use()

    # Instantiate the same base network as what was trained.
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    gen.to_gpu(device=0)

    # Then load the saved best parameters into the instance.
    serializers.load_npz(npz_file, gen)
    print("Loaded generator")
    return gen


# TODO: this should properly create training/val partitions, and be careful not to mix augmented data with original data
#  (rotations or synethic images) for the validation/test set.
def prepare_data(data_dir, transforms, batch_size, rotate_flag, gen_flag, gen_folders, num_gen_images, gen_classes, folds=3, trunc_flag=False, threshold=0.3, out='test'):
    print("Preparing data...")

    # Modify this to use the absolute file path.
    count = 0
    labels = []
    filenames = glob2.glob(data_dir + '/**/*.png')
    
    # Exclude rotations (which have "_r" in their name).
    filenames = [name for name in filenames if not name.__contains__("_r")]
    num_original_images = len(filenames)

    # Conversion dictionary for string label name into int.
    labels_dict = {
        "extremely": 0,
        "late": 4,
        "moderate": 2,
        "term": 3,
        "very": 1
    }

    for filename in filenames: 
        # Convert from class name to int label.
        dirname = filename.split('/')[-2]  # directory name is the class
        label = labels_dict[dirname]
        labels.append(label)
        count += 1
        #print(filename + " " + str(label))

    ## Next, use stratified sampling to get k training/test folds 
    # Create the kfold object
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    dataloader_k_folds = []
    # These training/test indices are the same ones that were used in training the generators (same seed),
    # although this time the test set is needed.
    split_index = 0 # For indexing the corresponding generator.
    for train_indices, test_indices in kfold.split(filenames, labels):
        # Convert to np arrays for list indexing.
        filenames = np.array(filenames)
        labels = np.array(labels)

        # Get train/test filenames and labels by the train/test indices.
        train_filenames = filenames[train_indices]
        train_labels = labels[train_indices]

        # This test set remains untouched and the dataloader can be created now.
        test_filenames = filenames[test_indices]
        test_labels = labels[test_indices]
        test_data = GestationalDataset(test_filenames, test_labels, transforms)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)

        print("Rotations?: %s | Generated images?: %s" % (str(rotate_flag), str(gen_flag)))
        # If specified, we now augment the training data. First we add in rotations/augmentations to the training data if specified.
        if rotate_flag:
            print("Adding rotations.")
            # Prepare rotations
            rot_files, rot_labels = prepare_rotations(train_filenames, labels_dict)
            # Add rotations and their labels to training filenames/labels.
            train_filenames = np.concatenate([train_filenames, rot_files])
            train_labels = np.concatenate([train_labels, rot_labels])
    
        if gen_flag:
            print("Adding generated images.")
            # Load generator for this split.
            gen = load_generator(gen_folders[split_index])
            gen_names = []
            gen_labels = []
            gen_data_dict = {}
            # For each class that is specified to augment
            for c in gen_classes:
                # Generate an amount of images
                for i in range(num_gen_images):
                    #TODO: Check if values are stored by reference/value.
                    img_name = "split_%i_class_%i_generated_image_%i" % (split_index, c, i)
                   
                    # Add id/label to combine with filename list.
                    gen_names.append(img_name)
                    gen_labels.append(c)

                    # Generate the image, convert it to PIL and add it to the img dictionary.
                    img = gen_image_with_condition(gen, c, trunc_flag, threshold)

                    # Make it same type as PIL  
                    pil_img = Image.fromarray(img)
                    pil_img.save(out + '/images/' + img_name + '.png') # test
                   
                    # Add it to the dictionary.
                    gen_data_dict[img_name] = pil_img
                   

            # Concatenate with original training data.
            train_filenames = np.concatenate([train_filenames, np.array(gen_names)])
            train_labels = np.concatenate([train_labels, np.array(gen_labels)])

        else:
            gen_data_dict = None

        # Debug
        #print(train_filenames)
        #print(train_labels)
        print("Number of training samples in split %i: %i" % (split_index, len(train_filenames)))
       
     
        # Prepare the data and dataloader.
        train_data = GestationalDataset(train_filenames, train_labels, transforms, gen_data_dict)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)

        # Save the train/test dataloaders of this k fold split.
        dataloaders_dict = {
            'train': train_loader,
            'test': test_loader
        }
        dataloader_k_folds.append(dataloaders_dict)

        # Increment index
        split_index += 1

    return num_original_images, dataloader_k_folds

# Prepare rotations based on the training samples for addressing the class imbalance.
def prepare_rotations(train_data, classes):
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
    return np.array(rotations), np.array(rotations_labels)                      

  


# Define a helper function to initialize the resnet model.
def initialize_model(num_classes, feature_extract, use_pretrained):
    resnet = models.resnet18(use_pretrained)

    if feature_extract:
        # Freeze previous layers.
        for params in resnet.parameters():
            params.requires_grad = False

    # Change the fully connected layer to have 5 classes.
    fc_input_dim = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features=fc_input_dim, out_features=num_classes)  # Create a new fc layer to replace it.

    # Network input size should be 224x224
    input_size = 224

    return resnet, input_size


# Define a helper function train the model.
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs):
    start_time = time.time()

    # Store loss and accuracies from each epoch here.
    acc_history = []
    loss_history = []

    # Initialize variables to keep track of the best model weights and accuracy (default is the input model).
    best_model_weights = copy.deepcopy(model.state_dict())  # Essentially a clone of the model (like Java obj clone).
    best_model_accuracy = 0.0

    model.train() # Set model mode to training.

    # Iterate over all epochs to train the model.
    for epoch in range(num_epochs):
        print("Epoch %i/%i" % (epoch, num_epochs - 1))
        print("-" * 10)  # Horizontal rule.

        # Each epoch has a training and validation/test phase.
        # Note that some layers behave differently during training/testing like batchnorm/dropout. E.g. batchnorm
        # uses a running average mean/std acquired from the training mode, rather than a batch mean/std.
       

        # Keep track of running loss and correct predictions across the dataset in this epoch.
        running_loss = 0.0
        running_corrects = 0

        # Iterate over all batches of images in the training dataloader. 
        for inputs, labels in dataloaders['train']:  # phase index loads train or test data.
            # Move images/label tensors to specified device (ideally GPU is available).
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Using with [stmt] 'cleans up' the stmt after the indentation ends.
            # If it is the training phase, then gradient calculation is enabled for training.
            # This ends after the with - it becomes disabled.
            # set_grad_enabled takes a boolean - so it is only active if True.
            with torch.set_grad_enabled(True):
                # Perform forward pass of the network.
                outputs = model(inputs)
                # Criterion calculates the loss for the batch given the output predictions and actual labels.
                loss = criterion(outputs, labels)


                # Note that for 5 classes, then the output is a 1x5 tensor, and for a batch size n, then there are
                # n rows of 5. Taking the max of each row corresponds to taking the highest probability class - the
                # prediction for that input.
                _, predictions = torch.max(outputs, 1)

                # Do backpropagation and an optimizer step.
                # Accumulates gradient of each weight (each has its own .grad attribute).
                loss.backward()
                # Perform a parameter/weight update based on this current gradient (in .grad).
                optimizer.step()

            # Update running statistics. Note that inputs has size N x C x H x W, where N is batch size,
            # C is channels (3 for RGB).
            running_loss += loss.item() * inputs.size(0)  # Extract loss value as a python float.
            running_corrects += torch.sum(predictions == labels.data)  # Sum correct predictions.

        # Calculate loss and accuracy for the epoch - divide by length of the dataset used (train/val).
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        print("%s Loss: %.4f Accuracy: %.4f" % ('Training', epoch_loss, epoch_acc))

        # Update the best model if the epoch accuracy is better than the current best.
        if epoch_acc > best_model_accuracy:
            best_model_accuracy = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())  # Clone.

        # Record epoch accuracy and loss
        acc_history.append(epoch_acc)
        loss_history.append(epoch_loss)
        # Newline before next epoch.
        print()

    # End of training
    end_time = time.time() - start_time
    print("Training finished after %.0fm %.0fs" % (end_time // 60, end_time % 60))
    print("Best training accuracy: %.4f" % best_model_accuracy)

    # Load the best model weights and return them with the validation accuracy history.
    model.load_state_dict(best_model_weights)

    # Use best model weights for calculating predictions on the test set.
    num_classes = 5
    print("Calculating predictions on test set")

    # Prepare predictions and label tensors for storing batch results
    test_pred = torch.zeros(0, dtype=torch.long, device=device)
    test_labels = torch.zeros(0, dtype=torch.long, device=device)

    # Set to eval mode
    model.eval()
    with torch.no_grad(): # Do not update weights
        # Iterate through validation batches
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get batch predictions
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            # Concatenate to overall results - torch.cat concatenates a sequence of tensors
            #print(test_pred)
            #print(predictions.view(-1))
            test_pred = torch.cat([test_pred, predictions.view(-1)])
            test_labels = torch.cat([test_labels, labels.view(-1)])
    ##print(test_pred)
    #print(test_labels)

    # # TODO: remove this Create the matrix
    # matrix = confusion_matrix(test_labels.cpu().numpy(), test_pred.cpu().numpy())
    # print("Confusion matrix: ")
    # print(matrix)

    # # Get accuracy per class
    # class_accuracy = 100 * matrix.diagonal()/matrix.sum(1)
    # print("Accuracy per class: ")
    # print(class_accuracy)

    return model, acc_history, loss_history, best_model_accuracy, test_pred, test_labels

# Allows the use of boolean command line args e.g. --flag True. Converts input string to boolean value.
def bool_arg(arg):
    if arg.lower() in ['true', 't', '1']:
        return True
    elif arg.lower() in ['false', 'f', '0']:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Specify dataset as command line argument.
    parser.add_argument("--dataset", type=str, default="gestational_face")
    parser.add_argument("--generated_dir", type=str, default='../FreezeD/projection/models_to_eval', help='directory of generator model folders for augmentation')
    parser.add_argument("--num_gen_images", type=int, default=10, help='number of images to generate per class for augmentation')
    parser.add_argument("--augment_gen", type=bool_arg, default=False, help='include generated image augmentation? true/false')
    parser.add_argument("--rotations", type=bool_arg, default=False, help='include rotation augmentation? true/false')
    parser.add_argument("--output_name", type=str, help='identifier for output files')
    parser.add_argument("--pretrained", type=bool_arg, default=True, help="True to use resnet18 pretrained on ImageNet, False from scratch")
    parser.add_argument("--num_epochs", type=int, default=100)



    # Data directory, cs/home/psyajhi/data/ will contain all assembled augmented datasets.
    args = parser.parse_args()
    data_dir = "/cs/home/psyajhi/data/" + args.dataset
    gen_dir = args.generated_dir

    # Create output directories
    out = './resnet_output/' + args.output_name
    if not os.path.exists(out):
        os.mkdir(out)
        os.mkdir(out + '/images')

    # Define the classes to augment with generated data - cannot easily pass an array as command line argument so
    # prepare that here. Some generators may have experienced mode collapse in the minority classes, so exclude these
    # from augmentation. (Extremely 0, Very 1, Moderate 2, Term 3, Late 4)
    if "face" or "ear" in args.dataset:
        # gen classes is... etc for ear/foot
        gen_classes = [2, 3, 4]
    else:
        # Foot
        gen_classes = [0, 1, 2, 3, 4]

    num_gen_images = args.num_gen_images

    # Get the specific generator folders based on the dataset used and generator directory.
    # Folders have the format: gen_dir/split_i_dataset
    gen_folders = sorted(glob2.glob(gen_dir + '/split_*_' + args.dataset))
    print(gen_folders)
    
    # Include rotations/synthetic data?
    rotate_flag = args.rotations
    gen_flag = args.augment_gen

    # Use truncation trick?
    trunc_flag = False
    threshold = 0.3

    # 5 classes; extremely, late, moderate, term, very.
    num_classes = 5

    # Batch size - depends on memory available.
    batch_size = 8

    # Epochs to train.
    epochs = args.num_epochs

    # Cross Validation folds
    folds = 3

    # Determines whether the previous layers of the resnet are trained.
    # True means it will be used as a fixed feature extractor - it won't be trained,
    # so model parameters in these layers will not require gradients to be calculated.
    # Should only be true if use_pretrained is true.
    feature_extract = args.pretrained
    use_pretrained = args.pretrained

    # Prepare data transforms for normalization.
    # Since augmentation will be prepared outside this script, no need for separate train/test transforms
    # Resnet requires 224x224 inputs, but images are 128x128. Use zero padding equally on each border.
    input_size = 224
    pad_length = (input_size - 128) // 2
    data_transforms = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Pad(pad_length, fill=0),
        transforms.ToTensor(),  # Take out if error
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # The pretrained model used these values.
    ])

    # Prepare the dataloaders. This returns a list with a dataloader dictionary for each partition.
    # Each dictionary has a 'train' and 'test' dataloader.
    num_original_images, dataloader_dicts = prepare_data(data_dir, data_transforms, batch_size, rotate_flag, 
                                                            gen_flag, gen_folders, num_gen_images, gen_classes, folds,
                                                            trunc_flag, threshold, out)

    # For storing the predictions/labels of the test set in each CV partition.
    test_pred_list = []
    test_labels_list =[]
    best_model_accuracy_list = []

    # Train and evaluate on the dataset across the cross validation partitions.
    for i in range(folds):

        print("Running CV iteration %i" % i)
        # Create model to train.
        model, input_size = initialize_model(num_classes, feature_extract, use_pretrained)

        # Use GPU if available.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Send model to GPU.
        model = model.to(device)

        # Create Optimizer to only update desired parameters/weights.
        # Assemble the parameters to be updated in this run to input to the optimizer.
        weights_to_train = model.parameters()
        print("Weights to train:")
        if feature_extract:
            weights_to_train = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weights_to_train.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        # Create optimizer
        optimizer = optim.SGD(weights_to_train, lr=0.001, momentum=0.9)

        # Set loss function (Cross entropy loss)
        criterion = nn.CrossEntropyLoss()

        print("Begin training")
        model, acc_history, loss_history, best_model_accuracy, test_pred, test_labels = train_model(model, dataloader_dicts[i], criterion, optimizer,
                                device, num_epochs=epochs)

        # Store test predictions/label for this partition.
        test_pred_list.append(test_pred)
        test_labels_list.append(test_labels)
        best_model_accuracy_list.append(best_model_accuracy)

    # Calculate accuracy of the model per splt, calculate average and s.d (this can help show variation between them.)
    acc_per_fold = []
    for i in range(folds):
        acc_per_fold.append((torch.sum(test_pred_list[i] == test_labels_list[i])/len(test_pred_list[i])).cpu().numpy())
    avg_test_acc = np.mean(acc_per_fold)
    std_test_acc = np.std(acc_per_fold)
        

    # Calculate confusion matrix based on all predictions made during cross validation.
    # Concatenate to overall results - torch.cat concatenates a sequence of tensors
    test_pred = torch.cat(test_pred_list)
    test_labels = torch.cat(test_labels_list)
    print(test_pred)
    print(test_labels)

    y_pred = test_pred.cpu().numpy()
    y_labels = test_labels.cpu().numpy()

    # Create the matrix
    matrix = confusion_matrix(y_labels, y_pred)
    print("Confusion matrix: ")
    print(matrix)

    # class labels for sklearn functions
    plot_labels = ['extremely', 'very', 'moderate', 'term', 'late']

    # Sum all predictions in the matrix. This sum should be equal to the number of images in the original training set,
    # as they all get featured in a test fold once through cross validation. This is to confirm implementation is correct.
    m_total = np.sum(matrix, axis=(0, 1))
    print("Total predictions: %i" % m_total)
    if num_original_images != m_total:
        exit("ERROR: Total predictions in confusion matrix is not equal to the number of original training samples.")


    # Get accuracy per class
    class_accuracy = 100 * matrix.diagonal()/matrix.sum(1)
    print("Accuracy per class: ")
    print(class_accuracy)

    # Get overall accuracy, precision and recall.
    # classification_report(true_labels, predictions)

    report = classification_report(y_labels, y_pred, target_names=plot_labels)
    print(report)

    
    
    # Save result
    with open(out + '_results.txt', "w") as f:
        f.write("Mean test accuracy per fold:\n")
        f.write(str(avg_test_acc))
        f.write("Standard deviation of test accuracy per fold: \n")
        f.write(str(std_test_acc))

        f.write("----- Confusion matrix results --------\n")
        f.write("Overall Accuracy: ")
        f.write(str(np.sum(y_labels == y_pred)/len(y_labels)) + "\n")
        f.write("Accuracy per class: \n")
        f.write(str(class_accuracy)+ "\n")
        f.write("Confusion matrix: \n")
        f.write(str(matrix) + "\n")
        f.write("Classification report: \n")
        f.write(report)

    # Save confusion matrix plot
    cm_display = ConfusionMatrixDisplay(matrix, display_labels=plot_labels).plot(cmap='Greys', colorbar=False)
    plt.savefig(out + '_confusion_matrix.png', bbox_inches='tight')



