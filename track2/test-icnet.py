import numpy as np
import os
from keras.models import load_model
from tqdm import tqdm
import tifffile
import cv2
import glob
from icnet import model_icnet
import argparse
from copy import deepcopy

#GPU="0" # default GPU
GPU="-1" # no GPU

NUM_CATEGORIES = 5

def sequential_to_las_labels(seq_labels):
    labels = deepcopy(seq_labels)
    labels[:] = 65
    labels[seq_labels == 0] = 2     # ground
    labels[seq_labels == 1] = 5     # trees
    labels[seq_labels == 2] = 6     # building roof
    labels[seq_labels == 3] = 9     # water
    labels[seq_labels == 4] = 17    # bridge / elevated road
    return labels

# convert category value image to RGB color image
def category_to_color(category_image):

    # define colors
    # color table is here: https://www.rapidtables.com/web/color/RGB_Color.html
    colors = []
    colors.append((165,42,42))      # 0  brown (ground)
    colors.append((0,128,0))        # 1  green (trees)
    colors.append((255,0,0))        # 2  red (buildings)
    colors.append((0,0,255))        # 3  blue (water)
    colors.append((128,128,128))    # 4  gray (elevated road)
    colors.append((0,0,0))          # 6  black (other)

    # convert categories to color image
    rows = category_image.shape[0]
    cols = category_image.shape[1]
    categories = category_image.astype(np.uint8)
    categories = np.reshape(categories, [rows, cols])
    rgb_image = cv2.cvtColor(categories,cv2.COLOR_GRAY2RGB)
    for i in range(cols):
        for j in range(rows):
            rgb_image[j,i,:] = colors[categories[j,i]]
    return rgb_image

if __name__=="__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('test_folder', type=str)    
    parser.add_argument('output_folder', type=str)        
    parser.add_argument('model_file', type=str)        
    args = parser.parse_args()

    # load the model
    height = 1024
    width = 1024
    bands = 3
    print(height, width, bands)
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU
    model = model_icnet.build_icnet(height, width, bands, NUM_CATEGORIES+1, weights_path=args.model_file, train=False)

    # predict semantic segmentation for all images in folder
    files = glob.glob(args.test_folder + '*LEFT_RGB.tif')
    nfiles = len(files)
    print('Number of files = ', nfiles)
    for i in tqdm(range(nfiles)):
        name = files[i]
        pos = name.find('LEFT_RGB')
        left_name = name
        name = os.path.basename(name)
        pos = name.find('LEFT_RGB')
        cls_name = args.output_folder + name[0:pos] + 'LEFT_CLS.tif'
        viz_name = args.output_folder + name[0:pos] + 'SEGMENTATION_RGB.tif'
        img = tifffile.imread(left_name)
        img = np.expand_dims(img,axis=0)
        img = (img - 127.5)/255.0
        seg = np.argmax(model.predict(img)[0,:,:,0:NUM_CATEGORIES],axis=2)
        # save RGB version of image for visual inspection
        tifffile.imsave(viz_name, category_to_color(seg))
        # save with LAS classification labels for metric analysis
        seg = sequential_to_las_labels(seg)
        tifffile.imsave(cls_name, seg, compress=6)
