import numpy as np
import os
from keras.models import load_model
from keras.applications import imagenet_utils
from tqdm import tqdm
import tifffile
import cv2
import glob
from densemapnet import densemapnet
import argparse

NO_DATA = -999.0
COMPLETENESS_THRESHOLD_PIXELS = 3.0

#GPU="0" # default GPU
GPU="-1" # no GPU

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
    settings = densemapnet.Settings()
    settings.xdim = width
    settings.ydim = height
    settings.channels = bands
    settings.nopadding = True
    settings.dropout_test = False
    settings.model_weights = args.model_file
    network = densemapnet.DenseMapNet(settings=settings)
    model = network.build_model()

    # predict disparities for all images in folder
    files = glob.glob(args.test_folder + '*LEFT_RGB.tif')
    nfiles = len(files)
    print('Number of files = ', nfiles)
    for i in tqdm(range(nfiles)):

        name = files[i]
        pos = name.find('LEFT_RGB')
        left_name = name
        right_name = name[0:pos] + 'RIGHT_RGB.tif'
        name = os.path.basename(name)
        pos = name.find('LEFT_RGB')
        dsp_name = args.output_folder + name[0:pos] + 'LEFT_DSP.tif'
        viz_name = args.output_folder + name[0:pos] + 'STEREO_GRAY.tif'
        left = tifffile.imread(left_name)
        right = tifffile.imread(right_name)
        left = np.expand_dims(left,axis=0)
        right = np.expand_dims(right,axis=0)

        # scale image values to [0,1]
        left = (left - 127.5)/255.0
        right = (right - 127.5)/255.0

        disp = model.predict([left, right])[0,:,:,:]
        tifffile.imsave(dsp_name, disp, compress=6)

        # save grayscale version of image for visual inspection
        disp = disp - disp.min()
        disp = ((disp / disp.max()) * 255.0).astype(np.uint8)
        disp = cv2.cvtColor(disp,cv2.COLOR_GRAY2RGB)
        tifffile.imsave(viz_name, disp, compress=6)

