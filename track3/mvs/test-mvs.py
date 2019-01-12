# This is a simple baseline semantic MVS algorithm using IC-Net and DenseMapNet
# to check that the US3D metadata is correct and to demonstrate
# epipolar rectification, RPC projections, UTM conversions, and triangulation.
# This example also shows how to read image metadata from the IMD files to
# help guide image pair selection.


# Code for estimating the Fundamental matrix was adapted from the following OpenCV python tutorial:
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html


# Code for SGBM stereo matching was adapted from Timotheos Samartzidis's blog:
# http://timosam.com/python_opencv_depthimage
# which elaborates on the following OpenCV tutorial:
# https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/disparity_filtering.cpp
# The weighted least squares filtering method used is based on the following paper:
# Min, Dongbo and Choi, Sunghwan and Lu, Jiangbo and Ham, Bumsub and Sohn, Kwanghoon and Do, Minh N,
# "Fast global image smoothing based on weighted least squares," IEEE Transactions on Image Processing, 2014.


# Epipolar rectification code was adapted from the following demo by Julien Rebetez:
# https://github.com/julienr/cvscripts/blob/master/rectification/rectification_demo.py
# Copyright (2012) Julien Rebetez <julien@fhtagn.net>. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this list 
#      of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice, this list of 
#      conditions and the following disclaimer in the documentation and/or other materials 
#      provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE FREEBSD PROJECT ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE FREEBSD PROJECT OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tifffile
import glob
import cv2
import numpy as np
import numpy.linalg as la
import cv2 as cv
from osgeo import gdal
import os
import math
from skimage import img_as_ubyte
from copy import deepcopy
import epipolar
from rpc import RPC
from utm import *
from model_icnet import build_icnet
from densemapnet import DenseMapNet
from densemapnet import Settings

# USE_SGM = False
USE_SGM = True
DMAX_SEARCH = 128  # max disparity range for search
# GPU = "-1"         # no GPU
GPU = "0"  # default GPU
NO_DATA = -999.0
NUM_CATEGORIES = 5


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


class Predictor(object):

    # initialize
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU
        self.height = 1024
        self.width = 1024
        self.bands = 3
        self.num_categories = NUM_CATEGORIES

    def build_seg_model(self, weights_file):
        print('building segmentation model')
        self.seg_model = build_icnet(self.height, self.width, self.bands, self.num_categories + 1,
                                     weights_path=weights_file, train=False)

    def build_stereo_model(self, weights_file):
        print('building stereo model')
        settings = Settings()
        settings.xdim = self.width
        settings.ydim = self.height
        settings.channels = self.bands
        settings.nopadding = True
        settings.dropout_test = False
        settings.model_weights = weights_file
        network = DenseMapNet(settings=settings)
        self.stereo_model = network.build_model()

    def predict_stereo(self, left, right):
        left = np.expand_dims(left, axis=0)
        right = np.expand_dims(right, axis=0)
        left = (left - 127.5) / 255.0
        right = (right - 127.5) / 255.0
        disp = self.stereo_model.predict([left, right])[0, :, :, 0]
        return disp

    def predict_semantics(self, img):
        img = np.expand_dims(img, axis=0)
        img = (img - 127.5) / 255.0
        cats = self.seg_model.predict(img)[0, :, :, 0:self.num_categories]
        seg = np.argmax(cats, axis=2)
        return seg


def sgbm(rimg1, rimg2):
    # run SGM stereo matching with weighted least squares filtering
    print('Running SGBM stereo matcher...')
    rimg1 = cv2.cvtColor(rimg1, cv2.COLOR_BGR2GRAY)
    rimg2 = cv2.cvtColor(rimg2, cv2.COLOR_BGR2GRAY)
    maxd = DMAX_SEARCH
    print('MAXD = ', maxd)
    window_size = 5
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-maxd,
        numDisparities=maxd * 2,
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


# read IMD file with sensor azimuth and elevation values
def read_imd(name):
    file = open(name, 'r')
    lines = file.readlines()
    for j in range(len(lines)):
        pos = lines[j].find('meanSatAz')
        if pos != -1:
            last = lines[j].find(';') - 1
            az = float(lines[j][pos + 11:last])
        pos = lines[j].find('meanSatEl')
        if pos != -1:
            last = lines[j].find(';') - 1
            el = float(lines[j][pos + 11:last])
    return az, el


# compute convergence angle given two sets of azimuths and elevations
def convergence_angle(az1, az2, el1, el2):
    cosd = math.sin(el1) * math.sin(el2) + math.cos(el1) * math.cos(el2) * math.cos(az1 - az2)
    d = math.degrees(math.acos(cosd))
    return d


# find virtual image correspondences given known RPC coefficients for both images
def match_rpc(rpc1, rpc2, rows, columns, x_km=1.0, z_km=0.5, num_samples=100):
    # get UTM zone
    clat, clon, cheight = rpc1.approximate_wgs84()
    easting, northing, zone_number, zone_letter = wgs84_to_utm(clat, clon)

    # sample local world coordinates around the center coordinate
    print('finding virtual xyz correspondences...')
    np.random.seed(0)
    dlat = dlon = (x_km / 2.0) / 111.0
    dheight = (z_km / 2.0) * 1000.0
    lat = np.random.uniform(clat - dlat, clat + dlat, num_samples)
    lon = np.random.uniform(clon - dlon, clon + dlon, num_samples)
    z = np.random.uniform(cheight - dheight, cheight + dheight, num_samples)

    # project into both images
    i1, j1 = rpc1.forward_array(lon, lat, z)
    i1 = np.int32(np.round(i1))
    j1 = np.int32(np.round(j1))
    i2, j2 = rpc2.forward_array(lon, lat, z)
    i2 = np.int32(np.round(i2))
    j2 = np.int32(np.round(j2))

    # remove invalid image coordinates
    keep = (i1 > 0) & (i1 < columns - 1) & (j1 > 0) & (j1 < rows - 1)
    lat = lon[keep]
    lon = lon[keep]
    i1 = i1[keep]
    j1 = j1[keep]
    i2 = i2[keep]
    j2 = j2[keep]
    keep = (i2 > 0) & (i2 < columns - 1) & (j2 > 0) & (j2 < rows - 1)
    lat = lon[keep]
    lon = lon[keep]
    i1 = i1[keep]
    j1 = j1[keep]
    i2 = i2[keep]
    j2 = j2[keep]

    print(np.asarray(i1).shape)

    count = np.asarray(i1).shape[0]
    pts1 = np.asarray([(i1, j1)])
    pts1 = pts1[0, :, :].transpose()
    pts2 = np.asarray([(i2, j2)])
    pts2 = pts2[0, :, :].transpose()
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    print('Points: ', len(pts1))
    print('Fundamental matrix = ')
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    print(F)
    return F, pts1, pts2


# get epipolar rectification matrices
def get_epipolar_rectification_matrices(img1, x1, img2, x2, K, d, F):
    imsize = (img1.shape[1], img1.shape[0])
    H1, H2, rms, max_error = epipolar.rectify_uncalibrated(x1, x2, F, imsize)
    rH = la.inv(K).dot(H1).dot(K)
    lH = la.inv(K).dot(H2).dot(K)
    return rH, lH


# get max y parallax using corresponding image points
def get_y_parallax(x1, x2, rH, lH, imsize):
    i1 = x1[0, :]
    i2 = x2[0, :]
    j1 = x1[1, :]
    j2 = x2[1, :]
    ydim = imsize[0]
    xdim = imsize[1]

    # project left image coordinates into left epipolar coordinates
    uv1 = np.array((i1, j1, np.ones(i1.shape[0])))
    xyw = np.matmul(rH, uv1)
    i1 = xyw[0, :] / xyw[2, :]
    j1 = xyw[1, :] / xyw[2, :]
    i1 = np.int32(np.round(i1))
    j1 = np.int32(np.round(j1))
    keep = (i1 >= 0) & (i1 < xdim - 1) & (j1 >= 0) & (j1 < ydim - 1)
    i1 = i1[keep]
    j1 = j1[keep]
    i2 = i2[keep]
    j2 = j2[keep]

    # project right image coordinates into right epipolar coordinates
    uv1 = np.array((i2, j2, np.ones(i2.shape[0])))
    xyw = np.matmul(lH, uv1)
    i2 = xyw[0, :] / xyw[2, :]
    j2 = xyw[1, :] / xyw[2, :]
    i2 = np.int32(np.round(i2))
    j2 = np.int32(np.round(j2))
    keep = (i2 >= 0) & (i2 < xdim - 1) & (j2 >= 0) & (j2 < ydim - 1)
    i1 = i1[keep]
    j1 = j1[keep]
    i2 = i2[keep]
    j2 = j2[keep]

    # get max y parallax
    max_yparallax = np.max(np.abs(j1 - j2))
    print('Max abs. y-parallax = ', max_yparallax)
    return max_yparallax


# rectify an image pair based on the Fundamental matrix
def rectify_images(img1, x1, img2, x2, K, d, F, shearing=False):
    imsize = (img1.shape[1], img1.shape[0])
    H1, H2, rms, max_error = epipolar.rectify_uncalibrated(x1, x2, F, imsize)
    if shearing:
        S = epipolar.rectify_shearing(H1, H2, imsize)
        H1 = S.dot(H1)
    rH = la.inv(K).dot(H1).dot(K)
    lH = la.inv(K).dot(H2).dot(K)

    # check for y parallax
    max_yparallax = get_y_parallax(x1, x2, rH, lH, imsize)

    # TODO: lRect or rRect for img1/img2 ??
    map1x, map1y = cv2.initUndistortRectifyMap(K, d, rH, K, imsize,
                                               cv.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K, d, lH, K, imsize,
                                               cv.CV_16SC2)

    # Convert the images to RGBA (add an axis with 4 values)
    img1 = np.tile(img1[:, :, np.newaxis], [1, 1, 4])
    img1[:, :, 3] = 255
    img2 = np.tile(img2[:, :, np.newaxis], [1, 1, 4])
    img2[:, :, 3] = 255

    rimg1 = cv2.remap(img1, map1x, map1y,
                      interpolation=cv.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))
    rimg2 = cv2.remap(img2, map2x, map2y,
                      interpolation=cv.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))

    # Put a red background on the invalid values
    # TODO: Return a mask for valid/invalid values
    # TODO: There is aliasing happening on the images border. We should
    # invalidate a margin around the border so we're sure we have only valid
    # pixels
    rimg1[rimg1[:, :, 3] == 0, :] = (255, 0, 0, 255)
    rimg2[rimg2[:, :, 3] == 0, :] = (255, 0, 0, 255)

    return rimg1, rimg2, rms, max_error, lH, rH, max_yparallax


# rectify an image pair based on the Fundamental matrix
def rectify_images_rgb(img1, x1, img2, x2, K, d, F, shearing=False):
    imsize = (img1.shape[1], img1.shape[0])
    H1, H2, rms, max_error = epipolar.rectify_uncalibrated(x1, x2, F, imsize)
    if shearing:
        S = epipolar.rectify_shearing(H1, H2, imsize)
        H1 = S.dot(H1)
    rH = la.inv(K).dot(H1).dot(K)
    lH = la.inv(K).dot(H2).dot(K)

    # TODO: lRect or rRect for img1/img2 ??
    map1x, map1y = cv2.initUndistortRectifyMap(K, d, rH, K, imsize, cv.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K, d, lH, K, imsize, cv.CV_16SC2)
    rimg1 = cv2.remap(img1, map1x, map1y,
                      interpolation=cv.INTER_CUBIC,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0))
    rimg2 = cv2.remap(img2, map2x, map2y,
                      interpolation=cv.INTER_CUBIC,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0))

    return rimg1, rimg2, rms, max_error


# rectify a floating point image pair based on the Fundamental matrix
# use this for XYZ images
def rectify_images_float(img1, x1, img2, x2, K, d, F, shearing=False):
    imsize = (img1.shape[1], img1.shape[0])
    H1, H2, rms, max_error = epipolar.rectify_uncalibrated(x1, x2, F, imsize)
    if shearing:
        S = epipolar.rectify_shearing(H1, H2, imsize)
        H1 = S.dot(H1)
    rH = la.inv(K).dot(H1).dot(K)
    lH = la.inv(K).dot(H2).dot(K)
    map1x, map1y = cv2.initUndistortRectifyMap(K, d, rH, K, imsize, cv.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K, d, lH, K, imsize, cv.CV_16SC2)

    rimg1 = cv2.remap(img1, map1x, map1y,
                      interpolation=cv.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))
    rimg2 = cv2.remap(img2, map2x, map2y,
                      interpolation=cv.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))

    return rimg1, rimg2


# get NITF metadata that we embedded in the GeoTIFF header
def get_image_metadata(img_name):
    dataset = gdal.Open(img_name, gdal.GA_ReadOnly)
    metadata = dataset.GetMetadata()
    rpc_data = dataset.GetMetadata('RPC')
    date_time = metadata['NITF_IDATIM']
    year = int(date_time[0:4])
    month = int(date_time[4:6])
    day = int(date_time[6:8])
    return metadata, month, day


# epipolar rectify an image pair and save the results
def stereo_to_xyz(img1_name, img2_name, predictor):
    # read rgb images as pan
    img1 = cv2.imread(img1_name, 0)
    img2 = cv2.imread(img2_name, 0)

    # read rgb images
    rgb_img1 = tifffile.imread(img1_name)
    rgb_img2 = tifffile.imread(img2_name)

    # get RPC metadata for both images
    rpc1 = RPC(img1_name)
    rpc2 = RPC(img2_name)

    # find corresponding points
    rows = img1.shape[0]
    cols = img1.shape[1]
    F, pts1, pts2 = match_rpc(rpc1, rpc2, rows, cols)

    # transpose the point matrices
    x1 = pts1.T
    x2 = pts2.T

    # set camera matrix to identity and distortion to zero
    d = None
    K = np.identity(3);

    # rectify the images
    rimg1, rimg2, rms, max_error, lH, rH, max_yparallax = rectify_images(img1, x1, img2, x2, K, d, F, shearing=False)
    print('F Matrix Residual RMS Error = ', rms, ' pixels')
    print('F Matrix Residual Max Error = ', max_error, ' pixels')

    # skip this pair if residual error is too large
    # this is not a good indicator of failure
    if rms > 1.0:
        return None, None, None, None, None

    # skip this pair if max y parallax is too large
    if max_yparallax > 2.0:
        return None, None, None, None, None

    # crop down to the center half of the images
    rows = img1.shape[0]
    cols = img1.shape[1]
    row_offset = int(rows / 4)
    col_offset = int(cols / 4)

    # rectify the RGB versions of the images and crop to center
    rgb_rimg1, rgb_rimg2, rgb_rms, rgb_max_error = rectify_images_rgb(rgb_img1, x1, rgb_img2, x2, K, d, F,
                                                                      shearing=False)
    rimg1 = rgb_rimg1[row_offset:rows - row_offset, col_offset:cols - col_offset, :]
    rimg2 = rgb_rimg2[row_offset:rows - row_offset, col_offset:cols - col_offset, :]

    # run DenseMapNet to get stereo disparities
    # and run IC-Net to get semantic labels
    if USE_SGM:
        disparity = sgbm(rimg1, rimg2)
    else:
        disparity = predictor.predict_stereo(rimg1, rimg2)
    #        disparity = predictor.predict_stereo_wls(rimg1, rimg2)
    seg_rimg1 = predictor.predict_semantics(rimg1)

    # check for invalid disparity predictions
    rows = rimg1.shape[0]
    cols = rimg1.shape[1]
    valid = np.ones((rows, cols), dtype=bool)
    valid[disparity < -DMAX_SEARCH] = 0
    valid[disparity > DMAX_SEARCH] = 0
    disparity[disparity < -DMAX_SEARCH] = -DMAX_SEARCH
    disparity[disparity > DMAX_SEARCH] = DMAX_SEARCH

    print('Min disparity found: ', disparity.min())
    print('Max disparity found: ', disparity.max())

    # create a grayscale disparity image
    disparity_image = (disparity + DMAX_SEARCH) / (DMAX_SEARCH * 2.0)
    disparity_image[disparity_image < 0.0] = 0.0
    disparity_image = img_as_ubyte(disparity_image)

    # create a color segmentation image
    cls_image = category_to_color(seg_rimg1)

    # get left epipolar image coordinates and right coordinates from disparities
    # add offsets from image cropping back to image coordinates
    # convert coordinates to original images using homographies from rectification
    rows = rimg1.shape[0]
    cols = rimg1.shape[1]
    left_rows, left_cols = np.mgrid[row_offset:rows + row_offset, col_offset:cols + col_offset]
    right_cols = deepcopy(left_cols) - disparity
    right_rows = deepcopy(left_rows)

    #    valid[right_cols < 0] = 0
    #    valid[right_cols > cols-1] = 0
    valid[right_cols < col_offset] = 0
    valid[right_cols > cols + col_offset - 1] = 0

    # left_rows = left_rows[:,DMAX_SEARCH:cols-DMAX_SEARCH]
    # left_cols = left_cols[:,DMAX_SEARCH:cols-DMAX_SEARCH]
    # right_rows = right_rows[:,DMAX_SEARCH:cols-DMAX_SEARCH]
    # right_cols = right_cols[:,DMAX_SEARCH:cols-DMAX_SEARCH]
    left_rows = left_rows.ravel()
    left_cols = left_cols.ravel()
    right_rows = right_rows.ravel()
    right_cols = right_cols.ravel()
    num = len(left_cols)
    print('left cols = ', num)
    uv1 = np.array((left_cols, left_rows, np.ones(num)))
    print(uv1.shape)
    print(rH.shape)
    xyw = np.matmul(np.linalg.inv(rH), uv1)
    left_cols = xyw[0] / xyw[2]
    left_rows = xyw[1] / xyw[2]
    uv2 = np.array((right_cols, right_rows, np.ones(num)))
    xyw = np.matmul(np.linalg.inv(lH), uv2)
    right_cols = xyw[0] / xyw[2]
    right_rows = xyw[1] / xyw[2]

    # get left segmentation labels for valid disparities
    # left_seg = seg_rimg1[:,DMAX_SEARCH:cols-DMAX_SEARCH]
    left_seg = seg_rimg1
    print('left_seg shape = ', left_seg.shape)
    left_seg = left_seg.ravel()

    # crop and ravel valid point list
    print('valid shape = ', valid.shape)
    # valid = valid[:,DMAX_SEARCH:cols-DMAX_SEARCH]
    valid = valid.ravel()
    print('valid shape = ', valid.shape)

    # approximate RPCs with local 3x4 matrices
    # use center coordinate of one of the images for reference
    clat, clon, zc = rpc1.approximate_wgs84()
    xc, yc, zone_number, zone_letter = wgs84_to_utm(clat, clon)
    R1, rms1, ic1, jc1 = rpc1.to_matrix(clat, clon, zc)
    R2, rms2, ic2, jc2 = rpc2.to_matrix(clat, clon, zc)

    # triangulate to compute XYZ coordinates for each pixel
    print('Triangulating...')
    points1 = np.array((left_cols - ic1, left_rows - jc1))
    points2 = np.array((right_cols - ic2, right_rows - jc2))
    xyz = cv2.triangulatePoints(R1, R2, points1, points2)
    xyz /= xyz[3]
    xyz = xyz[0:3, valid]
    xyz[0, :] += xc
    xyz[1, :] += yc
    xyz[2, :] += zc
    xyz = np.transpose(xyz)
    print(zone_number, zone_letter)

    # add cls to the xyz array
    xyzc = np.zeros((xyz.shape[0], xyz.shape[1] + 1))
    xyzc[:, 0:3] = xyz
    left_seg = left_seg[valid]
    xyzc[:, 3] = left_seg
    return xyzc, rimg1, rimg2, disparity_image, cls_image


# convert point cloud to surface
# heights are stored in decimeters
def xyz_to_dsm(xyz, easting, northing, pixels, gsd):
    dsm = np.zeros((pixels, pixels), dtype=np.float32)
    cls = np.zeros((pixels, pixels), dtype=np.uint8)
    x = np.uint16(np.round((xyz[:, 0] - easting) / gsd))
    y = np.uint16(np.round((xyz[:, 1] - northing) / gsd))
    z = xyz[:, 2]
    c = xyz[:, 3]
    dsm[:, :] = NO_DATA
    cls[:, :] = NUM_CATEGORIES
    keep = (x >= 0) & (x < pixels - 1) & (y >= 0) & (y < pixels - 1)
    x = x[keep]
    y = y[keep]
    z = z[keep]
    c = c[keep]
    dsm[pixels - 1 - y, x] = z
    cls[pixels - 1 - y, x] = c
    return dsm, cls


# convert category value image to RGB color image
def category_to_color(category_image):
    # define colors
    # color table is here: https://www.rapidtables.com/web/color/RGB_Color.html
    colors = []
    colors.append((165, 42, 42))  # 0  brown (ground)
    colors.append((0, 128, 0))  # 1  green (trees)
    colors.append((255, 0, 0))  # 2  red (buildings)
    colors.append((0, 0, 255))  # 3  blue (water)
    colors.append((128, 128, 128))  # 4  gray (elevated road)
    colors.append((0, 0, 0))  # 5  black (other)

    # convert categories to color image
    rows = category_image.shape[0]
    cols = category_image.shape[1]
    categories = category_image.astype(np.uint8)
    categories = np.reshape(categories, [rows, cols])
    rgb_image = cv2.cvtColor(categories, cv2.COLOR_GRAY2RGB)
    for i in range(cols):
        for j in range(rows):
            rgb_image[j, i, :] = colors[categories[j, i]]
    return rgb_image


def get_most_frequent_category(cat_array):
    cats = np.zeros((NUM_CATEGORIES))
    for i in range(NUM_CATEGORIES):
        cats[i] = len(cat_array[cat_array == i])
    return np.argmax(cats)


# merge a folder of DSM TIF images
def merge_dsm_tifs(folder, count, outname):
    # loop on all input DSM images
    dsm_images = []
    cls_images = []
    for i in range(count):
        name = folder + '{:03d}'.format(i) + '_dsm.tif'
        if os.path.isfile(name):
            next_image = tifffile.imread(name)
            dsm_images.append(next_image)
        name = folder + '{:03d}'.format(i) + '_cls.tif'
        if os.path.isfile(name):
            next_image = tifffile.imread(name)
            cls_images.append(next_image)

    # compute median value for each pixel
    print('Length = ', len(dsm_images))
    dsm_images = np.array(dsm_images)
    cls_images = np.array(cls_images)
    print(dsm_images.shape)
    count = dsm_images.shape[0]
    ydim = dsm_images.shape[1]
    xdim = dsm_images.shape[2]
    median_dsm = np.zeros((ydim, xdim), dtype=np.float32)
    median_cls = np.zeros((ydim, xdim), dtype=np.uint8)
    for i in range(ydim):
        for j in range(xdim):
            pixel = dsm_images[:, i, j]
            pixel = pixel[pixel != NO_DATA]
            count = pixel.shape[0]
            if (count > 0):
                median_dsm[i, j] = np.median(pixel)
            else:
                median_dsm[i, j] = NO_DATA
            pixel = cls_images[:, i, j]
            median_cls[i, j] = get_most_frequent_category(pixel)

    # fill any remaining voids with the max value
    median_dsm[median_dsm == NO_DATA] = median_dsm.max()  # set to max height

    # convert CLS image to LAS conventions
    median_cls = sequential_to_las_labels(median_cls)

    # write median images
    tifffile.imsave(outname + '_DSM.tif', median_dsm)
    tifffile.imsave(outname + '_CLS.tif', median_cls)

    # write median images as uint8 tif for visualization
    median_u8_image = median_dsm - np.min(median_dsm)
    median_u8_image = np.uint8(np.round((median_u8_image / np.max(median_u8_image)) * 255))
    tifffile.imsave(outname + '_stereo_rgb.tif', median_u8_image)
    median_cls_rgb = las_to_sequential_labels(median_cls)
    median_cls_rgb = category_to_color(median_cls_rgb)
    tifffile.imsave(outname + '_segmentation_rgb.tif', median_cls_rgb)

    return median_dsm, median_cls


# main program to demonstrate a baseline MVS algorithm
if __name__ == "__main__":

    # select input and output folders
    nsites = 2
    site_strings = ['JAX', 'OMA']
    tile_numbers = [[163, 171, 208, 218, 266], [250, 284, 291, 333, 334]]  # JAX, then OMA VALIDATE tile numbers

    # Contains data provided by aptly named zips
    infolder = '../data/validate/Validate-Track3/'
    truthfolder = '../data/validate/Validate-Track3/'
    imdfolder = '../data/Metadata/Track3-Metadata'

    workingfolder = '../data/validate/Track3-Working/'
    outfolder = '../data/validate/Track3-Submission/'

    stereo_weights_file = '../track2/weights/181230-dfc2019.track2.densemapnet.weights.20-20.h5'
    seg_weights_file = '../track2/weights/190101-us3d.icnet.weights.18-3.h5'

    # initialize the models
    predictor = Predictor()
    predictor.build_stereo_model(stereo_weights_file)
    predictor.build_seg_model(seg_weights_file)

    # loop on all MVS sets
    for num in range(nsites):
        site_string = site_strings[num]
        for tile_number in tile_numbers[num]:

            # read tile bounds
            bounds_file = truthfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_DSM.txt'
            easting, northing, pixels, gsd = np.loadtxt(bounds_file)
            pixels = np.int16(pixels)

            print(infolder + site_string + '_' + '{:03d}'.format(tile_number) + '*_RGB*.tif')

            # get image file list
            files = glob.glob(infolder + site_string + '_' + '{:03d}'.format(tile_number) + '*_RGB*.tif')
            nfiles = len(files)
            print('Number of files = ', nfiles)

            # get azimuth and elevation angles for all images
            # also get semantic segmentation outputs for all images.
            azimuths = []
            elevations = []
            months = []
            for i in range(nfiles):
                # get image index
                pos = files[i].find('_RGB')
                ndx = int(files[i][pos - 3:pos])
                # get IMD metadata for this image
                imd_name = imdfolder + site_string + '/{:02d}'.format(ndx) + '.IMD'
                az, el = read_imd(imd_name)
                azimuths.append(az)
                elevations.append(el)
                meta, month, day = get_image_metadata(files[i])
                months.append(month)

            # get list of pairs to process
            pairs = []
            convergence = []
            distances = []
            for i in range(nfiles - 1):
                for j in range(i + 1, nfiles):
                    pair = [files[i], files[j]]
                    pairs.append(pair)
                    # get convergence angle
                    d = convergence_angle(azimuths[i], azimuths[j], elevations[i], elevations[j])
                    convergence.append(d)
                    # get time distance in months
                    dist = abs(months[i] - months[j])
                    dist = min(dist, 12 - dist)
                    distances.append(dist)
            npairs = len(pairs)
            print('Number of pairs = ', npairs)

            # sort list by convergence angle and limit number of pairs
            #            convergence = np.asarray(convergence)
            #            indices = convergence.argsort()
            distances = np.asarray(distances)
            indices = distances.argsort()
            pairs = [pairs[i] for i in indices]
            npairs = min(npairs, 50)
            print('Number of pairs = ', npairs)

            # process all pairs and save XYZ files in output folder
            good = 0
            for i in range(npairs):
                # run a single stereo pair
                xyz, rimg1, rimg2, disparity_image, cls_image = stereo_to_xyz(pairs[i][0], pairs[i][1], predictor)
                if xyz is not None:
                    # save a DSM for this stereo pair
                    print('Converting point cloud to DSM...')
                    dsm, cls = xyz_to_dsm(xyz, easting, northing, pixels, gsd)
                    fname = workingfolder + '{:03d}'.format(good) + '_dsm.tif'
                    tifffile.imsave(fname, dsm)
                    fname = workingfolder + '{:03d}'.format(good) + '_cls.tif'
                    tifffile.imsave(fname, cls)
                    fname = workingfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_{:03d}'.format(
                        good) + '_dsm.tif'
                    tifffile.imsave(fname, dsm)
                    # save epipolar images
                    fname = workingfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_{:03d}'.format(
                        good) + '_left.tif'
                    tifffile.imsave(fname, rimg1)
                    fname = workingfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_{:03d}'.format(
                        good) + '_right.tif'
                    tifffile.imsave(fname, rimg2)
                    fname = workingfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_{:03d}'.format(
                        good) + '_disparity.tif'
                    tifffile.imsave(fname, disparity_image)
                    fname = workingfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_{:03d}'.format(
                        good) + '_segmentation.tif'
                    tifffile.imsave(fname, cls_image)
                    good = good + 1
            print('Fraction Good = ', good / float(npairs))

            # merge the DSM files and write the final MVS product
            if good > 0:
                outname = outfolder + site_string + '_' + '{:03d}'.format(tile_number)
                median_dsm, median_cls = merge_dsm_tifs(workingfolder, good, outname)

                # get normalization factors
                minval = np.min(median_dsm)
                median_u8_image = median_dsm - minval
                maxval = np.max(median_u8_image)
                median_u8_image = np.uint8(np.round((median_u8_image / maxval) * 255))

                # read the truth and write a pretty picture for comparison
                truth_file = truthfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_DSM.tif'
                if os.path.isfile(truth_file):
                    truth_image = tifffile.imread(truth_file)
                    truth_u8_image = truth_image - minval
                    truth_u8_image = np.round((truth_u8_image / maxval) * 255)
                    truth_u8_image[truth_u8_image > 255] = 255
                    truth_u8_image[truth_u8_image < 0] = 0
                    truth_u8_image = np.uint8(truth_u8_image)
                    tifffile.imsave(outname + '_stereo_truth.tif', truth_u8_image)

                # read the truth CLS file and write a pretty picture for comparison
                truth_file = truthfolder + site_string + '_' + '{:03d}'.format(tile_number) + '_CLS.tif'
                if os.path.isfile(truth_file):
                    truth_cls = tifffile.imread(truth_file)
                    truth_cls = las_to_sequential_labels(truth_cls)
                    tifffile.imsave(outname + '_segmentation_truth.tif', category_to_color(truth_cls))
