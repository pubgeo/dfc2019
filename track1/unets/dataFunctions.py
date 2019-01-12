__author__ = 'jhuapl'
__version__ = 0.1

import os
import numpy as np
from glob import glob
import tifffile

from keras.applications import imagenet_utils
from keras.utils import to_categorical

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness
)


def parse_args(argv, params):
    """
    Parses input argument to determine if the code trains or tests a model and whether or not 
        the task is semantic segmentation or single-view depth prediction.
    :param argv: input arguments from main
    :param params: input parameters from params.py
    :return: modes of operation
    """
    
    isTrain = None
    
    argOptions1 = '1st argument options: train, test.'
    argOptions2 = '2nd argument options: semantic, single-view.'
    noArgStr = 'No arguments provided.' 
    incorrectArgStr = 'Incorrect argument provided.'
    insufficientArgStr = 'Not enough arguments provided.'
    exampleUsageStr = 'python runBaseline.py train semantic'
    
    try:
        trainStr = argv[1].lower()
    except:
        raise ValueError('%s %s %s' % (noArgStr,argOptions1,exampleUsageStr))
    
    try:
        modeStr = argv[2].lower()
    except:
        raise ValueError('%s %s %s' % (insufficientArgStr,argOptions2,exampleUsageStr))
        
    if trainStr == 'train':
        isTrain = True
    elif trainStr == 'test':
        isTrain = False
    else:
        raise ValueError('%s %s %s' % (incorrectArgStr,argOptions1,exampleUsageStr))
 
    if modeStr == 'semantic':
        mode = params.SEMANTIC_MODE
    elif modeStr == 'single-view':
        mode = params.SINGLEVIEW_MODE
    else:
        raise ValueError('%s %s %s' % (incorrectArgStr,argOptions2,exampleUsageStr))
        
    if (mode==params.SEMANTIC_MODE) and (params.NUM_CATEGORIES==1) and (params.SEMANTIC_LOSS=='categorical_crossentropy'):
        _=input('Warning: NUM_CATEGORIES is 1, but loss is not binary_crossentropy. You should probably change this in params.py, but press enter to continue.')
        
    return isTrain,mode


def get_image_paths(params, isTest=None):
    """
    Generates a list semantic ground truth files, which are used to load RGB and 
        depth files later with string replacements (i.e., only use image data that has semantic ground truth)
    :param params: input parameters from params.py
    :param isTest: determines whether or not to get image files for training or testing
    :return: list of paths to use for training
    """
    
    if isTest:
        return glob(os.path.join(params.TEST_DIR, '*%s*.%s' % (params.IMG_FILE_STR,params.IMG_FILE_EXT)))
    else:
        img_paths = []
        wildcard_image = '*%s.%s' % (params.CLASS_FILE_STR, params.LABEL_FILE_EXT)
        glob_path = os.path.join(params.LABEL_DIR, wildcard_image)
        curr_paths = glob(glob_path)
        for currPath in curr_paths:
            image_name = os.path.split(currPath)[-1]
            image_name = image_name.replace(params.CLASS_FILE_STR, params.IMG_FILE_STR)
            image_name = image_name.replace(params.LABEL_FILE_EXT, params.IMG_FILE_EXT)
            img_paths.append(os.path.join(params.TRAIN_DIR, image_name))
    return img_paths


def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    if imgPath.endswith('.tif'):
        img = tifffile.imread(imgPath)
    else:
        raise ValueError('Install pillow and uncomment line in load_img')
#        img = np.array(Image.open(imgPath))
    return img


def image_augmentation(currImg, labelMask):
    """
    Apply random image augmentations
    :param currImg: original image
    :param labelMask: original ground truth
    :return: post-augmentation image and ground truth data
    """
    aug = Compose([VerticalFlip(p=0.5),              
            RandomRotate90(p=0.5),HorizontalFlip(p=0.5),Transpose(p=0.5)])

    augmented = aug(image=currImg, mask=labelMask)
    imageMedium = augmented['image']
    labelMedium = augmented['mask']
    
    return imageMedium,labelMedium


def image_batch_preprocess(imgBatch, params, meanVals):
    """
    Apply preprocessing operations to the image data that also need to be applied during inference
    :param imgBatch: numpy array containing image data
    :param params: input parameters from params.py
    :param meanVals: used for mean subtraction if non-rgb imagery
    :return: numpy array containing preprocessed image data
    """
    if params.NUM_CHANNELS==3:
        imgBatch  = imagenet_utils.preprocess_input(imgBatch)
        imgBatch = imgBatch / 255.0
    else:
        for c in range(params.NUM_CATEGORIES):
            imgBatch[:,:,:,c] -= meanVals[c]
        imgBatch = imgBatch / params.MAX_VAL
    return imgBatch


def get_label_mask(labelPath, params, mode):
    """
    Loads the ground truth image (semantic or depth)
    :param labelPath: Path to the ground truth file (CLS or AGL file)
    :param params: input parameters from params.py
    :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
    :return: numpy array containing ground truth
    """
    currLabel = load_img(labelPath)
    if mode == params.SINGLEVIEW_MODE:
        currLabel[np.isnan(currLabel)] = params.IGNORE_VALUE
    elif mode == params.SEMANTIC_MODE:
        if params.CONVERT_LABELS:
            currLabel = convert_labels(currLabel, params, toLasStandard=False)
        if params.NUM_CATEGORIES > 1:
            currLabel = to_categorical(currLabel, num_classes=params.NUM_CATEGORIES+1)
    return currLabel


def load_batch(inds, trainData, params, mode, meanVals=None):
    """
    Given the batch indices, load the images and ground truth (labels or depth data)
    :param inds: batch indices
    :param trainData: training paths of CLS files (string replacement to get RGB and depth files) and starting x,y pixel positions (can be      non-zero if blocking is set to happen in params.py)
    :param params: input parameters from params.py
    :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
    :param meanVals: used for mean subtraction if non-rgb imagery
    :return: numpy arrays for image and ground truth batch data
    """
    
    if params.BLOCK_IMAGES:
        batchShape = (params.BATCH_SZ, params.BLOCK_SZ[0], params.BLOCK_SZ[1])
    else:
        batchShape = (params.BATCH_SZ, params.IMG_SZ[0], params.IMG_SZ[1])
    
    imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS))
    
    numChannels = None
    if mode == params.SINGLEVIEW_MODE:
        numChannels = 1
        labelReplaceStr = params.DEPTH_FILE_STR
    elif mode == params.SEMANTIC_MODE:
        numChannels = params.NUM_CATEGORIES
        labelReplaceStr = params.CLASS_FILE_STR

    labelBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], numChannels))
    
    batchInd = 0
    for i in inds:
        currData = trainData[i]
        imgPath = currData[0]
        if params.LABEL_DIR != params.TRAIN_DIR:
            imageName = os.path.split(imgPath)[-1]
            if params.LABEL_FILE_EXT != params.IMG_FILE_EXT:
                imageName = imageName.replace('.'+params.IMG_FILE_EXT, '.'+params.LABEL_FILE_EXT)
            labelPath = os.path.join(params.LABEL_DIR, imageName.replace(params.IMG_FILE_STR, labelReplaceStr))
        else:
            labelPath = imgPath.replace(params.IMG_FILE_STR,labelReplaceStr)
        currImg = load_img(imgPath)
        currLabel = get_label_mask(labelPath, params, mode)
        
        rStart,cStart = currData[1:3]
        rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
        currImg = currImg[rStart:rEnd, cStart:cEnd, :]
        if mode == params.SINGLEVIEW_MODE:
            currLabel = currLabel[rStart:rEnd, cStart:cEnd]
        else:
            currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
        
        imageMedium,labelMedium = image_augmentation(currImg, currLabel)

        imgBatch[batchInd,:,:,:] = imageMedium
        if mode == params.SINGLEVIEW_MODE:
            labelBatch[batchInd,:,:,0] = labelMedium
        else:
            labelBatch[batchInd,:,:,:] = labelMedium[:,:,:params.NUM_CATEGORIES]
            
        batchInd += 1

    imgBatch  = image_batch_preprocess(imgBatch, params, meanVals)

    return imgBatch,labelBatch


def get_batch_inds(idx, params):
    """
    Given a list of indices (random sorting happens outside), break into batches of indices for training
    :param idx: list of indices to break
    :param params: input parameters from params.py
    :return: List where each entry contains batch indices to pass through at the current iteration
    """

    N = len(idx)
    batchInds = []
    idx0 = 0
    toProcess = True
    while toProcess:
        idx1 = idx0 + params.BATCH_SZ
        if idx1 > N:
            idx1 = N
            idx0 = idx1 - params.BATCH_SZ
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1
    return batchInds


def convert_labels(Lorig, params, toLasStandard=True):
    """
    Convert the labels from the original CLS file to consecutive integer values starting at 0
    :param Lorig: numpy array containing original labels
    :param params: input parameters from params.py
    :param toLasStandard: determines if labels are converted from the las standard labels to training labels
        or from training labels to the las standard
    :return: Numpy array containing the converted labels
    """
    L = Lorig.copy()
    if toLasStandard:
        labelMapping = params.LABEL_MAPPING_TRAIN2LAS
    else:
        labelMapping = params.LABEL_MAPPING_LAS2TRAIN
        
    for key,val in labelMapping.items():
        L[Lorig==key] = val
        
    return L


def get_blocks(params):
    """
    Create blocks using the image dimensions, block size and overlap.
    :param params: input parameters from params.py
    :return: List of start row/col indices of the blocks
    """
    blocks = []
    yEnd,xEnd = np.subtract(params.IMG_SZ, params.BLOCK_SZ)
    x = np.linspace(0, xEnd, np.ceil(xEnd/np.float(params.BLOCK_SZ[1]-params.BLOCK_MIN_OVERLAP))+1, endpoint=True).astype('int')
    y = np.linspace(0, yEnd, np.ceil(yEnd/np.float(params.BLOCK_SZ[0]-params.BLOCK_MIN_OVERLAP))+1, endpoint=True).astype('int')
    
    for currx in x:
        for curry in y:
            blocks.append((currx,curry))
            
    return blocks


def get_train_data(imgPaths, params):
    """
    Create training data containing image paths and block information. If the full image is being used
        then the start row/col values are always 0,0. 
    :param imgPaths: list of image paths to be used for training
    :param params: input parameters from params.py
    :return: List of training data with image paths and block information
    """
    blocks = get_blocks(params)
    trainData = []
    for imgPath in imgPaths:
        for block in blocks:
            trainData.append((imgPath,block[0],block[1]))
            
    return trainData
