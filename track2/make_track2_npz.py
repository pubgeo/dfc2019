# convert folders of images to npz train/validation sets
# for training DenseMapNet and ICNet models
from os.path import join
from pathlib import Path
from sys import stderr

import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
import tifffile
import glob

TRAIN_FRACTION = 0.95
MAX_IMAGES_PER_TRAIN_FILE = 200


def sequential_to_las_labels(seq_labels):
    labels = deepcopy(seq_labels)
    labels[:] = 65
    labels[seq_labels == 0] = 2  # ground
    labels[seq_labels == 1] = 5  # trees
    labels[seq_labels == 2] = 6  # building roof
    labels[seq_labels == 3] = 9  # water
    labels[seq_labels == 4] = 17  # bridge / elevated road
    return labels


def las_to_sequential_labels(las_labels):
    labels = deepcopy(las_labels)
    labels[:] = 5  # unlabeled
    labels[las_labels == 2] = 0  # ground
    labels[las_labels == 5] = 1  # trees
    labels[las_labels == 6] = 2  # building roof
    labels[las_labels == 9] = 3  # water
    labels[las_labels == 17] = 4  # bridge / elevated road
    return labels


# create npz files
def convert_files_to_npz(input_folder, out_folder, out_prefix):
    # get list of files
    files = glob.glob(input_folder + '*LEFT_RGB*.tif')
    num = len(files)
    print('Number of images = ', num)
    if num == 0:
        print("No matching files found", file=stderr)
        return

    # determine size of train and test sets
    train_fraction = TRAIN_FRACTION
    num_train = int(train_fraction * num)
    max_per_train = MAX_IMAGES_PER_TRAIN_FILE

    print('Number of training images = ', num_train)
    print('Number of validation images = ', num - num_train)

    # initialize lists and counters
    count = 0
    num_files = 0
    disparities = []
    lefts = []
    rights = []
    left_categories = []
    left_agls = []

    # Shuffle the file list
    indices = np.arange(num)
    np.random.seed(0)
    np.random.shuffle(indices)
    files = [files[i] for i in indices]

    # loop on all files
    for i in tqdm(range(num)):

        # get file names
        left_name = os.path.basename(files[i])
        start = left_name.find('LEFT_RGB')
        right_name = input_folder + left_name[0:start] + 'RIGHT_RGB.tif'
        left_agl_name = input_folder + left_name[0:start] + 'LEFT_AGL.tif'
        disparity_name = input_folder + left_name[0:start] + 'LEFT_DSP.tif'
        left_cls_name = input_folder + left_name[0:start] + 'LEFT_CLS.tif'
        left_name = input_folder + left_name

        # read files
        left = np.array(tifffile.imread(left_name))
        right = np.array(tifffile.imread(right_name))
        left_cls = np.array(tifffile.imread(left_cls_name))
        disparity = np.array(tifffile.imread(disparity_name))
        left_agl = np.array(tifffile.imread(left_agl_name))

        # convert LAS labels to sequential labeling scheme for training           
        left_labels = las_to_sequential_labels(left_cls)

        # add images to lists after confirming that all corresponding files exist
        lefts.append(left)
        rights.append(right)
        disparities.append(disparity)
        left_categories.append(left_labels)
        left_agls.append(left_agl)

        # update the image counter
        count = count + 1

        # when counter gets too high, save new files
        if ((count >= max_per_train) and (i < num_train)) or (i == num_train - 1):

            # update the file counter
            num_files = num_files + 1

            # print counts for categories
            print(' ')
            print('Counts for train file ', num_files)
            cats = np.asarray(left_categories)
            max_category = cats.max()
            for j in range(max_category):
                print(j, ': ', len(cats[cats == j]))
            print('Writing files...')
            print(' ')

            # write the next training files if needed
            out_path = Path(out_folder)
            if not out_path.exists():
                out_path.mkdir()

            disparity_name = join(out_path, out_prefix + '.train.disparity.' + '{:1d}'.format(num_files) + '.npz')
            left_name = join(out_path, out_prefix + '.train.left.' + '{:1d}'.format(num_files) + '.npz')
            right_name = join(out_path, out_prefix + '.train.right.' + '{:1d}'.format(num_files) + '.npz')
            left_cat_name = join(out_path, out_prefix + '.train.left_label.' + '{:1d}'.format(num_files) + '.npz')
            left_agl_name = join(out_path, out_prefix + '.train.left_agl.' + '{:1d}'.format(num_files) + '.npz')
            np.savez_compressed(disparity_name, disparities)
            np.savez_compressed(left_name, lefts)
            np.savez_compressed(right_name, rights)
            np.savez_compressed(left_cat_name, left_categories)
            np.savez_compressed(left_agl_name, left_agls)

            # reset counter and all lists
            count = 0
            disparities = []
            lefts = []
            rights = []
            left_categories = []
            left_agls = []

    # print counts for categories
    print(' ')
    print('Counts for validation file')
    cats = np.asarray(left_categories)
    max_category = cats.max()
    for j in range(max_category):
        print(j, ': ', len(cats[cats == j]))
    print('Writing files...')
    print(' ')

    out_path = Path(out_folder)
    if not out_path.exists():
        out_path.mkdir()

    # write the validation set
    print('Writing validation files')
    print('Number of validation samples = ', len(disparities))
    disparity_name = join(out_path, out_prefix + '.test.disparity.npz')
    left_name = join(out_path, out_prefix + '.test.left.npz')
    right_name = join(out_path, out_prefix + '.test.right.npz')
    left_cat_name = join(out_path, out_prefix + '.test.left_label.npz')
    left_agl_name = join(out_path, out_prefix + '.test.left_agl.npz')
    np.savez_compressed(disparity_name, disparities)
    np.savez_compressed(left_name, lefts)
    np.savez_compressed(right_name, rights)
    np.savez_compressed(left_cat_name, left_categories)
    np.savez_compressed(left_agl_name, left_agls)


if __name__ == '__main__':
    in_folder = '../data/train/Track2/'
    output_folder = '../data/train/track2_npz'
    output_prefix = 'dfc2019.track2'
    convert_files_to_npz(in_folder, output_folder, output_prefix)
