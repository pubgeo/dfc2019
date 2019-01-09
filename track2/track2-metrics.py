# Metrics for track 2 - pairwise semantic stereo
import numpy as np
import tifffile
from glob import glob
import argparse
from tqdm import tqdm
from copy import deepcopy

NUM_CATEGORIES = 5
NO_DATA = -999.0
COMPLETENESS_THRESHOLD_PIXELS = 3.0


def las_to_sequential_labels(las_labels):
    labels = deepcopy(las_labels)
    labels[:] = 5  # unlabeled
    labels[las_labels == 2] = 0  # ground
    labels[las_labels == 5] = 1  # trees
    labels[las_labels == 6] = 2  # building roof
    labels[las_labels == 9] = 3  # water
    labels[las_labels == 17] = 4  # bridge / elevated road
    return labels


# compute miou-3 for a folder of submissions
def compute_metrics(test_folder, truth_folder):
    # get lists of files
    test_dsp_files = glob(test_folder + '*DSP*.tif')
    truth_dsp_files = glob(truth_folder + '*DSP*.tif')
    test_cls_files = glob(test_folder + '*CLS*.tif')
    truth_cls_files = glob(truth_folder + '*ClS*.tif')
    if len(test_dsp_files) != len(truth_dsp_files):
        print('Error: different number of submission and truth files')
        return None
    if len(test_cls_files) != len(truth_cls_files):
        print('Error: different number of submission and truth files')
        return None
    num_files = len(test_dsp_files)
    print('Number of files = ', num_files)
    if num_files == 0:
        print('Error: no files found')
        return None

    test_dsp_files.sort()
    truth_dsp_files.sort()
    test_cls_files.sort()
    truth_cls_files.sort()

    # loop on all files computing statistics
    tp = np.zeros(NUM_CATEGORIES)
    fp = np.zeros(NUM_CATEGORIES)
    fn = np.zeros(NUM_CATEGORIES)
    tp3 = np.zeros(NUM_CATEGORIES)
    fp3 = np.zeros(NUM_CATEGORIES)
    fn3 = np.zeros(NUM_CATEGORIES)
    for i in tqdm(range(num_files)):

        # load test and truth files
        test_cls = np.ravel(tifffile.imread(test_cls_files[i]))
        truth_cls = np.ravel(tifffile.imread(truth_cls_files[i]))
        test_dsp = np.ravel(tifffile.imread(test_dsp_files[i]))
        truth_dsp = np.ravel(tifffile.imread(truth_dsp_files[i]))

        # convert CLS values to sequential
        test_cls = las_to_sequential_labels(test_cls)
        truth_cls = las_to_sequential_labels(truth_cls)

        # accumulate statistics for IOU
        for cat in range(NUM_CATEGORIES):
            tp[cat] += ((test_cls == cat) & (truth_cls == cat) & (truth_cls < NUM_CATEGORIES)).sum()
            fp[cat] += ((test_cls == cat) & (truth_cls != cat) & (truth_cls < NUM_CATEGORIES)).sum()
            fn[cat] += ((test_cls != cat) & (truth_cls == cat) & (truth_cls < NUM_CATEGORIES)).sum()

        # accumulate statistics for IOU-3
        for cat in range(NUM_CATEGORIES):
            valid_disparity = (truth_dsp == NO_DATA) | (truth_cls == 3) | (
                        abs(test_dsp - truth_dsp) < COMPLETENESS_THRESHOLD_PIXELS)
            tp3[cat] += ((test_cls == cat) & (truth_cls == cat) & (truth_cls < NUM_CATEGORIES) & valid_disparity).sum()
            fp3[cat] += ((test_cls == cat) & (truth_cls != cat) & (truth_cls < NUM_CATEGORIES)).sum()
            fn3[cat] += ((test_cls != cat) & (truth_cls == cat) & (truth_cls < NUM_CATEGORIES)).sum()

    # compute IOU-3
    iou = np.divide(tp, tp + fp + fn)
    iou_3 = np.divide(tp3, tp3 + fp3 + fn3)
    print('IOU:  ', iou)
    print('mIOU: ', iou.mean())
    print('IOU-3:  ', iou_3)
    print('mIOU-3: ', iou_3.mean())
    return iou_3


# main
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('test_folder', type=str)
    parser.add_argument('truth_folder', type=str)
    args = parser.parse_args()
    iou3 = compute_metrics(args.test_folder, args.truth_folder)
