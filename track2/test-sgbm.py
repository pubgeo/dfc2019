
# Code for SGBM stereo matching was adapted from Timotheos Samartzidis's blog: http://timosam.com/python_opencv_depthimage
# which elaborates on the following OpenCV tutorial:
# https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/disparity_filtering.cpp
# The weighted least squares filtering method used is based on the following paper:
# Min, Dongbo and Choi, Sunghwan and Lu, Jiangbo and Ham, Bumsub and Sohn, Kwanghoon and Do, Minh N,
# "Fast global image smoothing based on weighted least squares," IEEE Transactions on Image Processing, 2014.


import numpy as np
import os
from tqdm import tqdm
import tifffile
import cv2
import glob
import argparse

DMAX_SEARCH = 128
NO_DATA = -999.0
COMPLETENESS_THRESHOLD_PIXELS = 3.0

# Semiglobal Block Matching (SGBM) with Weighted Least Squares (WLS) filtering 
def sgbm(rimg1, rimg2):
    rimg1 = cv2.cvtColor(rimg1, cv2.COLOR_RGB2GRAY)
    rimg2 = cv2.cvtColor(rimg2, cv2.COLOR_RGB2GRAY)
    maxd = DMAX_SEARCH
    window_size = 5
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-maxd,
        numDisparities=maxd*2,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    lmbda = 8000
    sigma = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(rimg1, rimg2)
    dispr = right_matcher.compute(rimg2, rimg1)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, rimg1, None, dispr) / 16.0
    return disparity

if __name__=="__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('test_folder', type=str)    
    parser.add_argument('output_folder', type=str)        
    args = parser.parse_args()

    # run SGBM on images in test folder and write to output folder
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
        left = tifffile.imread(left_name)
        right = tifffile.imread(right_name)
        disp = sgbm(left, right)
        tifffile.imsave(dsp_name, disp, compress=6)
        disp = disp - disp.min()
        disp = ((disp / disp.max()) * 255.0).astype(np.uint8)
        disp = cv2.cvtColor(disp,cv2.COLOR_GRAY2RGB)
        tifffile.imsave(args.output_folder + name[0:pos] + 'LEFT_VIZ.tif', disp, compress=6)

