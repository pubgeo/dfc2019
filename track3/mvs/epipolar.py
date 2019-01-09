# Minor modifications added by Myron Brown, 2018
# - Original code adapted from https://github.com/julienr/cvscripts/blob/master/rectification/rectification_demo.py
# - Copied the license language into the code to avoid it getting misplaced
# - Moved homg functions into this file
# - Added computation of point to line RMS error in rectify_uncalibrated
# - Updated for compatibility with most recent OpenCV library


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

import cv2
import cv2 as cv
import numpy as np
import numpy.linalg as la
import pylab as pl
import matplotlib.cm as cm
import math

def to_homg(x):
    """
    Transform x to homogeneous coordinates
    If X is MxN, returns an (M+1)xN array with ones on the last row

    >>> to_homg(np.array([[1, 2, 3], [1, 2, 3]], dtype=float))
    array([[ 1.,  2.,  3.],
           [ 1.,  2.,  3.],
           [ 1.,  1.,  1.]])
    >>> to_homg(np.array([[1], [2]]))
    array([[ 1.],
           [ 2.],
           [ 1.]])
    >>> to_homg([1, 2])
    array([1, 2, 1])
    """
    if hasattr(x, 'shape') and len(x.shape) > 1:
        return np.r_[x, np.ones((1,x.shape[1]))]
    else:
        return np.r_[x, 1]


def from_homg(x):
    """
    Transform homogeneous x to non-homogeneous coordinates
    If X is MxN, returns an (M-1)xN array that will contain nan when for
    columns where the last row was 0

    >>> from_homg(np.array([[1, 2, 3],
    ...                     [4, 5, 0]], dtype=float))
    array([[ 0.25,  0.4 ,   nan]])
    >>> from_homg(np.array([1, 5], dtype=float))
    array([ 0.2])
    >>> from_homg([1, 5, 0])
    array([ nan,  nan])
    >>> from_homg((1, 4, 0.5))
    array([ 2.,  8.])
    """
    if hasattr(x, 'shape') and len(x.shape) > 1:
        #valid = np.nonzero(x[-1,:])
        valid = x[-1,:] != 0
        result = np.empty((x.shape[0]-1, x.shape[1]), dtype=float)
        result[:,valid] = x[:-1,valid] / x[-1, valid]
        result[:,~valid] = np.nan
        return result
    else:
        if x[-1] == 0:
            result = np.empty(len(x)-1, dtype=float)
            result[:] = np.nan
            return result
        else:
            return np.array(x[:-1]) / x[-1]

def show_rectified_images(rimg1, rimg2):
    ax = pl.subplot(121)
    pl.imshow(rimg1, cmap=cm.gray)

    # Hack to get the lines span on the left image
    # http://stackoverflow.com/questions/6146290/plotting-a-line-over-several-graphs
    for i in range(1, rimg1.shape[0], int(rimg1.shape[0]/20)):
        pl.axhline(y=i, color='g', xmin=0, xmax=1.2, clip_on=False);

    pl.subplot(122)
    pl.imshow(rimg2, cmap=cm.gray)
    for i in range(1, rimg1.shape[0], int(rimg1.shape[0]/20)):
        pl.axhline(y=i, color='g');


def epipolar_lines(x1, F12):
    """
    Compute epipolar lines on second image corresponding to a list of
    points on first image
    As in cv.ComputecorrespondEpilines, line coefficient are normalized
    so that a^2 + b^2 = 1
    """
    def norm_line(l):
      nu = l[0]*l[0] + l[1]*l[1]
      return l * (1./np.sqrt(nu) if nu > 0 else 1.)
    return np.array([norm_line(F12.dot(to_homg(x))) for x in x1.T],
                    dtype=float).T

def rectify_shearing(H1, H2, imsize):
    """Compute shearing transform than can be applied after the rectification
    transform to reduce distortion.
    See :
    http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
    "Computing rectifying homographies for stereo vision" by Loop & Zhang
    """
    w = imsize[0]
    h = imsize[1]

    a = ((w-1)/2., 0., 1.)
    b = (w-1., (h-1.)/2., 1.)
    c = ((w-1.)/2., h-1., 1.)
    d = (0., (h-1.)/2., 1.)

    ap = from_homg(H1.dot(a))
    bp = from_homg(H1.dot(b))
    cp = from_homg(H1.dot(c))
    dp = from_homg(H1.dot(d))

    x = bp - dp
    y = cp - ap

    k1 = (h*h*x[1]*x[1] + w*w*y[1]*y[1]) / (h*w*(x[1]*y[0] - x[0]*y[1]))
    k2 = (h*h*x[0]*x[1] + w*w*y[0]*y[1]) / (h*w*(x[0]*y[1] - x[1]*y[0]))

    if k1 < 0:
        k1 *= -1
        k2 *= -1

    return np.array([[k1, k2, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=float)

def rectify_uncalibrated(x1, x2, F, imsize, threshold=5):
    """
    Compute rectification homography for two images. This is based on
    algo 11.12.3 of HZ2
    This is also heavily inspired by cv::stereoRectifyUncalibrated

    Args:
        - imsize is (width, height)
    """
    U, W, V = la.svd(F)
    # Enforce rank 2 on fundamental matrix
    W[2] = 0
    W = np.diag(W)
    F = U.dot(W).dot(V)

    # Filter points based on their distance to epipolar lines
    if threshold > 0:
        lines1 = epipolar_lines(x1, F)
        lines2 = epipolar_lines(x2, F.T)

        def epi_threshold(i):
            return (abs(x1[0,i]*lines2[0,i] +
                        x1[1,i]*lines2[1,i] +
                        lines2[2,i]) <= threshold) and \
                    (abs(x2[0,i]*lines1[0,i] +
                         x2[1,i]*lines1[1,i] +
                         lines1[2,i]) <= threshold)

        inliers = filter(epi_threshold, range(x1.shape[1]))
    else:
        inliers = range(x1.shape[1])

    #print(inliers)
    inliers = list(inliers) # mzb

    assert len(inliers) > 0

    x1 = x1[:,inliers]
    x2 = x2[:,inliers]

    # mzb
    # Compute and report the residual RMS distance to epipolar lines.
    rms = 0.0
    max_error = 0.0
    for i in range(x1.shape[1]):
        e1 = abs(x1[0,i]*lines2[0,i] + x1[1,i]*lines2[1,i] + lines2[2,i])
        e2 = abs(x2[0,i]*lines1[0,i] + x2[1,i]*lines1[1,i] + lines1[2,i])
        rms = rms + (e1 + e2)/2.0
        max_error = max(max_error, e1)
        max_error = max(max_error, e2)
    rms = math.sqrt(rms / x1.shape[1])

    # HZ2 11.12.1 : Compute H = GRT where :
    # - T is a translation taking point x0 to the origin
    # - R is a rotation about the origin taking the epipole e' to (f,0,1)
    # - G is a mapping taking (f,0,1) to infinity

    # e2 is the left null vector of F (the one corresponding to the singular
    # value that is 0 => the third column of U)
    e2 = U[:,2]

    # TODO: They do this in OpenCV, not sure why
    if e2[2] < 0:
        e2 *= -1

    # Translation bringing the image center to the origin
    # FIXME: This is kind of stupid, but to get the same results as OpenCV,
    # use cv.Round function, which has a strange behaviour :
    # cv.Round(99.5) => 100
    # cv.Round(132.5) => 132
    cx = round((imsize[0]-1)*0.5) # mzb
    cy = round((imsize[1]-1)*0.5) # mzb

    T = np.array([[1, 0, -cx],
                  [0, 1, -cy],
                  [0, 0, 1]], dtype=float)

    e2 = T.dot(e2)
    mirror = e2[0] < 0

    # Compute rotation matrix R that should bring e2 to (f,0,1)
    # 2D norm of the epipole, avoid division by zero
    d = max(np.sqrt(e2[0]*e2[0] + e2[1]*e2[1]), 1e-7)
    alpha = e2[0]/d
    beta = e2[1]/d
    R = np.array([[alpha, beta, 0],
                  [-beta, alpha, 0],
                  [0, 0, 1]], dtype=float)

    e2 = R.dot(e2)

    # Compute G : mapping taking (f,0,1) to infinity
    invf = 0 if abs(e2[2]) < 1e-6*abs(e2[0]) else -e2[2]/e2[0]
    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [invf, 0, 1]], dtype=float)

    # Map the origin back to the center of the image
    iT = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0, 1]], dtype=float)

    H2 = iT.dot(G.dot(R.dot(T)))

    # HZ2 11.12.2 : Find matching projective transform H1 that minimize
    # leaste-square distance between reprojected points
    e2 = U[:,2]

    # TODO: They do this in OpenCV, not sure why
    if e2[2] < 0:
        e2 *= -1

    e2_x = np.array([[0, -e2[2], e2[1]],
                     [e2[2], 0, -e2[0]],
                     [-e2[1], e2[0], 0]], dtype=float)

    e2_111 = np.array([[e2[0], e2[0], e2[0]],
                       [e2[1], e2[1], e2[1]],
                       [e2[2], e2[2], e2[2]]], dtype=float)

    H0 = H2.dot(e2_x.dot(F) + e2_111)

    # Minimize \sum{(a*x_i + b*y_i + c - x'_i)^2} (HZ2 p.307)
    # Compute H1*x1 and H2*x2
    x1h = to_homg(x1)
    x2h = to_homg(x2)
    A = H0.dot(x1h).T
    # We want last (homogeneous) coordinate to be 1 (coefficient of c
    # in the equation)
    A = (A.T / A[:,2]).T # for some reason, A / A[:,2] doesn't work
    B = H2.dot(x2h)
    B = B / B[2,:] # from homogeneous
    B = B[0,:] # only interested in x coordinate

    X, _, _, _ = la.lstsq(A, B, rcond=None)

    # Build Ha (HZ2 eq. 11.20)
    Ha = np.array([[X[0], X[1], X[2]],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=float)

    H1 = Ha.dot(H0)

    if mirror:
        mm = np.array([[-1, 0, cx*2],
                       [0, -1, cy*2],
                       [0, 0, 1]], dtype=float)
        H1 = mm.dot(H1)
        H2 = mm.dot(H2)

    return H1, H2, rms, max_error
