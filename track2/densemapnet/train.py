# Minor modifications added by Myron Brown, 2018
# - Original code adapted from https://github.com/roatienza/densemapnet
# - Replaced binary cross-entropy with custom MSE and MAE loss functions
# - Added checks to ignore NO_DATA values and out of bounds truth disparities
# - Updated to directly regress disparity values which can be negative


# MIT License
# Copyright (c) 2018 Rowel Atienza
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc
from densemapnet import Settings
from densemapnet import DenseMapNet
from keras import objectives
from keras import backend as K
import tensorflow as tf
import tifffile

NO_DATA = -999.0

# MSE loss function that ignores NO_DATA
def _loss_mse_disparity(y_true, y_pred):
    out = K.square(y_pred - y_true)
    mask = tf.greater(y_true, NO_DATA)
    out = tf.boolean_mask(out, mask)
    return K.mean(out, axis=-1)

# MAE loss function that ignores NO_DATA
def _loss_mae_disparity(y_true, y_pred):
    out = K.abs(y_pred - y_true)
    mask = tf.greater(y_true, NO_DATA)
    out = tf.boolean_mask(out, mask)
    myloss = K.mean(out, axis=-1)
    return myloss

class Predictor(object):
    def __init__(self, settings=Settings()):
        self.settings = settings
        self.mkdir_images()
        self.pdir = settings.pdir
        self.get_max_disparity()
        self.load_test_data()
        self.load_mask()
        self.network  = None
        self.train_data_loaded = False
        if self.settings.epe:
            self.best_epe = self.settings.epe
        else:
            self.best_epe = 100.0

    def load_mask(self):
        if self.settings.mask:
            if self.settings.predict:
                filename = self.settings.dataset + "_complete.test.mask.npz"
                print("Loading... ", filename)
                self.test_mx = np.load(os.path.join(self.pdir, filename))['arr_0']
            else:
                filename = self.settings.dataset + ".test.mask.npz"
                print("Loading... ", filename)
                self.test_mx = np.load(os.path.join(self.pdir, filename))['arr_0']
                filename = self.settings.dataset + ".train.mask.1.npz"
                print("Loading... ", filename)
                self.train_mx = np.load(os.path.join(self.pdir, filename))['arr_0']
                shape = [-1, self.train_mx.shape[1], self.train_mx.shape[2], 1]
                self.train_mx = np.reshape(self.train_mx, shape)

            shape = [-1, self.test_mx.shape[1], self.test_mx.shape[2], 1]
            self.test_mx = np.reshape(self.test_mx, shape)

    def load_test_disparity(self):
        filename = self.settings.dataset + ".test.disparity.npz"
        print("Loading... ", filename)
        self.test_dx = np.load(os.path.join(self.pdir, filename))['arr_0']
        self.dmax =  max(self.dmax, np.amax(self.test_dx))
        self.dmin =  min(self.dmin, np.amin(self.test_dx))
        self.sum += np.sum(self.test_dx, axis=0)
        self.size += self.test_dx.shape[0]
        print("Max disparity on entire dataset: ", self.dmax)
        print("Min disparity on entire dataset: ", self.dmin)
        if self.settings.predict:
            filename = self.settings.dataset + "_complete.test.disparity.npz"
            print("Loading... ", filename)
            self.test_dx = np.load(os.path.join(self.pdir, filename))['arr_0']

        dim = self.test_dx.shape[1] * self.test_dx.shape[2]
        self.ave = np.sum(self.sum / self.size) / dim
        print("Ave disparity: ", self.ave)
        self.test_dx = self.test_dx.astype('float32')
        print("Scaled disparity max: ", np.amax(self.test_dx))
        print("Scaled disparity min: ", np.amin(self.test_dx))

        # set out of bounds disparities to NO_DATA
        nrows = self.test_dx.shape[1]
        ncols = self.test_dx.shape[2]
        xmin = 0
        xmax = ncols
        rows, cols = np.indices((nrows, ncols))
        mask = (cols + self.test_dx < 0)
        self.test_dx[mask] = NO_DATA
        mask = (cols + self.test_dx >= ncols)
        self.test_dx[mask] = NO_DATA

        shape = [-1, self.test_dx.shape[1], self.test_dx.shape[2], 1]
        self.test_dx = np.reshape(self.test_dx, shape)

    def get_max_disparity(self):
        self.dmax = 0
        self.dmin = 255
        self.sum = None
        self.size = 0
        count = self.settings.num_dataset + 1
        for i in range(1, count, 1):
            filename = self.settings.dataset + ".train.disparity.%d.npz" % i
            print("Loading... ", filename)
            self.train_dx = np.load(os.path.join(self.pdir, filename))['arr_0']
            self.dmax =  max(self.dmax, np.amax(self.train_dx))
            self.dmin =  min(self.dmin, np.amin(self.train_dx))
            if self.sum is None:
                self.sum = np.sum(self.train_dx, axis=0)
            else:
                self.sum += np.sum(self.train_dx, axis=0)
            self.size += self.train_dx.shape[0]
        self.load_test_disparity()

    def load_test_data(self):
        if self.settings.predict:
            filename = self.settings.dataset + "_complete.test.left.npz"
            print("Loading... ", filename)
            self.test_lx = np.load(os.path.join(self.pdir, filename))['arr_0']
            filename = self.settings.dataset + "_complete.test.right.npz"
            print("Loading... ", filename)
            self.test_rx = np.load(os.path.join(self.pdir, filename))['arr_0']
        else:
            filename = self.settings.dataset + ".test.left.npz"
            print("Loading... ", filename)
            self.test_lx = np.load(os.path.join(self.pdir, filename))['arr_0']
            filename = self.settings.dataset + ".test.right.npz"
            print("Loading... ", filename)
            self.test_rx = np.load(os.path.join(self.pdir, filename))['arr_0']
        self.channels = self.settings.channels = self.test_lx.shape[3]
        self.xdim = self.settings.xdim = self.test_lx.shape[2]
        self.ydim = self.settings.ydim = self.test_lx.shape[1]

        # scale image values to [0,1]
        self.test_lx = (self.test_lx - 127.5)/255.0
        self.test_rx = (self.test_rx - 127.5)/255.0

    def load_train_data(self, index):
        filename = self.settings.dataset + ".train.left.%d.npz" % index
        print("Loading... ", filename)
        self.train_lx = np.load(os.path.join(self.pdir, filename))['arr_0']

        filename = self.settings.dataset + ".train.right.%d.npz" % index
        print("Loading... ", filename)
        self.train_rx = np.load(os.path.join(self.pdir, filename))['arr_0']

        # scale image values to [0,1]
        self.train_lx = (self.train_lx - 127.5)/255.0
        self.train_rx = (self.train_rx - 127.5)/255.0

        filename = self.settings.dataset + ".train.disparity.%d.npz" % index
        print("Loading... ", filename)
        self.train_dx = np.load(os.path.join(self.pdir, filename))['arr_0']
        self.train_dx = self.train_dx.astype('float32')
        print("Scaled disparity max: ", np.amax(self.train_dx))
        print("Scaled disparity min: ", np.amin(self.train_dx))

        # set out of bounds disparities to NO_DATA
        nrows = self.train_dx.shape[1]
        ncols = self.train_dx.shape[2]
        xmin = 0
        xmax = ncols
        print(nrows)
        print(ncols)
        rows, cols = np.indices((nrows, ncols))
        mask = (cols + self.train_dx < 0)
        self.train_dx[mask] = NO_DATA
        mask = (cols + self.train_dx >= ncols)
        self.train_dx[mask] = NO_DATA

        shape =  [-1, self.train_dx.shape[1], self.train_dx.shape[2], 1]
        self.train_dx = np.reshape(self.train_dx, shape)
        self.channels = self.settings.channels = self.train_lx.shape[3]
        self.xdim = self.settings.xdim = self.train_lx.shape[2]
        self.ydim = self.settings.ydim = self.train_lx.shape[1]
        self.train_data_loaded = True

    def train_network(self):
        # if data is not split into multiple files
        if self.settings.num_dataset == 1:
            self.train_all()
            return

        # if multiple data files (only for "driving" data set) and not training
        if self.settings.notrain:
            self.train_batch()
            return

        # if multiple data files and training
        epochs = 1
        decay = 1e-6
        lr = 1e-3 + decay
        for i in range(400):
            lr = lr - decay
            print("Epoch: ", (i+1), " Learning rate: ", lr)
            self.train_batch(epochs=epochs, lr=lr, seq=(i+1))
            self.predict_disparity()


    def train_all(self, epochs=1000, lr=1e-3):
        checkdir = "checkpoint"
        try:
            os.mkdir(checkdir)
        except FileExistsError:
            print("Folder exists: ", checkdir)

        filename = self.settings.dataset
        filename += ".densemapnet.weights.{epoch:02d}.h5"
        filepath = os.path.join(checkdir, filename)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=False)
        predict_callback = LambdaCallback(on_epoch_end=lambda epoch,
                                          logs: self.predict_disparity())
        callbacks = [checkpoint, predict_callback]
        self.load_train_data(1)
        if self.network is None:
            self.network = DenseMapNet(settings=self.settings)
            self.model = self.network.build_model()

#        print("Using loss=mse_disparity on final conv layer and Adam")
        print("Using loss=mae_disparity on final conv layer and Adam")
        print(self.settings.lr)
#        self.model.compile(loss='mse',
#        self.model.compile(loss=_loss_mse_disparity,
        self.model.compile(loss=_loss_mae_disparity,
#        self.model.compile(loss=_loss_3pe_disparity,
#        optimizer=RMSprop(lr=lr, decay=1e-6))
        optimizer=Adam(lr=self.settings.lr))		
			
        if self.settings.model_weights:
            if self.settings.notrain:
                self.predict_disparity()
                return

        x = [self.train_lx, self.train_rx]
        self.model.fit(x,
                       self.train_dx,
                       epochs=epochs,
                       batch_size=self.settings.batch_size,
                       shuffle=True,
                       callbacks=callbacks)

    def train_batch(self, epochs=1, lr=1e-3, seq=1):
        count = self.settings.num_dataset + 1
        checkdir = "checkpoint"
        try:
            os.mkdir(checkdir)
        except FileExistsError:
            print("Folder exists: ", checkdir)

        is_model_compiled = False
            
        indexes = np.arange(1,count)
        np.random.shuffle(indexes)
        
        for i in indexes:
            filename = self.settings.dataset
            #filename += ".densemapnet.weights.{epoch:02d}.h5"
            filename += ".densemapnet.weights.%d-%d.h5" % (seq, i)
            filepath = os.path.join(checkdir, filename)
            checkpoint = ModelCheckpoint(filepath=filepath,
                                         save_weights_only=True,
                                         verbose=1,
                                         save_best_only=False)
            callbacks = [checkpoint]

            self.load_train_data(i)

            if self.network is None:
                self.network = DenseMapNet(settings=self.settings)
                self.model = self.network.build_model()
	
            if not is_model_compiled:
                print("Using loss=mae on final conv layer")
                self.model.compile(loss=_loss_mae_disparity, optimizer=Adam(lr=lr))	
                is_model_compiled = True

            if self.settings.model_weights:
                if self.settings.notrain:
                    self.predict_disparity()
                    return

            x = [self.train_lx, self.train_rx]
            self.model.fit(x,
                           self.train_dx,
                           epochs=epochs,
                           batch_size=self.settings.batch_size,
                           shuffle=True,
                           callbacks=callbacks)

    def mkdir_images(self):
        self.images_pdir = "images"
        pdir = self.images_pdir

        for dirname in ["train", "test"]:
            cdir = os.path.join(pdir, dirname)
            filepath = os.path.join(cdir, "left")
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(cdir, "right")
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(cdir, "disparity")
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(cdir, "prediction")
            os.makedirs(filepath, exist_ok=True)

    def get_epe(self, use_train_data=True, get_performance=False):
        if use_train_data:
            lx = self.train_lx
            rx = self.train_rx
            dx = self.train_dx
            if self.settings.mask:
                print("Using mask images")
                mx = self.train_mx
            print("Using train data... Size: ", lx.shape[0])
        else:
            lx = self.test_lx
            rx = self.test_rx
            dx = self.test_dx
            if self.settings.mask:
                print("Using mask images")
                mx = self.test_mx
            if self.settings.predict:
                print("Using complete data... Size: ", lx.shape[0])
            else:
                print("Using test data... Size: ", lx.shape[0])

        # sum of all errors (normalized)
        epe_total = 0
        my3pe_total = 0
        # count of images
        t = 0
        nsamples = lx.shape[0]
        elapsed_total = 0.0
        if self.settings.images:
            print("Saving images on folder...")
        for i in range(0, nsamples, 1):
            indexes = np.arange(i, i + 1)
            left_images = lx[indexes, :, :, : ]
            right_images = rx[indexes, :, :, : ]
            disparity_images = dx[indexes, :, :, : ]
            if self.settings.mask:
                mask_images = mx[indexes, :, :, : ]
            # measure the speed of prediction on the 10th sample to avoid variance
            if get_performance:
                start_time = time.time()
                predicted_disparity = self.model.predict([left_images, right_images])
                elapsed_total += (time.time() - start_time)
            else:
                predicted_disparity = self.model.predict([left_images, right_images])

            predicted = predicted_disparity[0, :, :, :]
            # if self.settings.mask:
            # ground_mask = np.ceil(mask_images[0, :, :, :])
            # predicted = np.multiply(predicted, ground_mask)

            ground = disparity_images[0, :, :, :]
            if self.settings.mask:
                ground_mask = mask_images[0, :, :, :]
                dim = np.count_nonzero(ground_mask)
                nz = np.nonzero(ground_mask)
                epe = predicted[nz] - ground[nz]
            else:
                dim = predicted.shape[0] * predicted.shape[1]
                epe = predicted - ground

            # filter out the invalid points.
            mask = ground != NO_DATA
            epe = epe[mask]
            dim = len(epe)

            # set invalid predictions to zero for output images.
            mask = (ground == NO_DATA)
            ground[mask] = 0

            # 3 pixel error for this image pair
            my3pe = np.sum(np.absolute(epe) < 3.0)
            my3pe = my3pe.astype('float32');
            my3pe = my3pe / dim
            my3pe_total += my3pe

            # normalized error on all pixels
            epe = np.sum(np.absolute(epe))
            epe = epe.astype('float32')
            epe = epe / dim
            epe_total += epe

            if (get_performance and self.settings.images) or ((i%1) == 0): 
                path = "test"
                if use_train_data:
                    path = "train"
                filepath  = os.path.join(self.images_pdir, path)
                left = os.path.join(filepath, "left")
                right = os.path.join(filepath, "right")
                disparity = os.path.join(filepath, "disparity")
                prediction = os.path.join(filepath, "prediction")
                filename = "%04d.png" % i
                tiffname = "%04d.tif" % i
                left = os.path.join(left, filename)
                # if left_images[0].shape[2] == 1:
                    # self.save_rgb_image(left_images[0], left)
                # else:
                    # plt.imsave(left, left_images[0])
                self.save_rgb_image(left_images[0], left)

                right = os.path.join(right, filename)
                # if right_images[0].shape[2] == 1:
                    # self.save_rgb_image(right_images[0], right)
                # else:
                    # plt.imsave(right, right_images[0])
                self.save_rgb_image(right_images[0], right)

                self.save_disparity_image(predicted, os.path.join(prediction, tiffname))
                self.save_disparity_image(ground, os.path.join(disparity, tiffname))

        # average endpoint error
        epe = epe_total / nsamples

        # average 3 pixel error
        tpe = 1.0 - (my3pe_total / nsamples)

        if self.settings.dataset == "kitti2015":
            epe = epe / 256.0
            print("KITTI 2015 EPE: ", epe)
        else:
            print("EPE: %0.2fpix" % epe)
            print("3PE: %0.2f" % tpe) # add 3 pixel error

        # report if best EPE
        if epe < self.best_epe:
            self.best_epe = epe
            print("------------------- BEST EPE : %f ---------------------" % epe)
            tmpdir = "tmp"
            try:
                os.mkdir(tmpdir)
            except FileExistsError:
                print("Folder exists: ", tmpdir)
            filename = open('tmp\\epe.txt', 'a')
            datetime = time.strftime("%H:%M:%S")
            filename.write("%s : LR: %f : %s EPE: %f 3PE: %f\n" % (datetime, self.settings.lr, self.settings.dataset, epe, tpe))
            filename.close()

        # speed in sec
        if get_performance:
            print("Speed: %0.4fsec" % (elapsed_total / nsamples))
            print("Speed: %0.4fHz" % (nsamples / elapsed_total))

    def save_rgb_image(self, image, filepath):
#        size = [image.shape[0], image.shape[1]]
        minval = np.amin(image)
        maxval = np.amax(image)
        image = (image - minval)/ (maxval - minval + 0.01)
        image *= 255
        image = image.astype(np.uint8)
#        image = np.reshape(image, size)
        tifffile.imsave(filepath, image)

    def save_disparity_image(self, image, filepath):
        size = [image.shape[0], image.shape[1]]
        minval = -self.dmax
        maxval = self.dmax
        image = (image - minval)/ (maxval - minval + 0.01)
        image[image < 0.0] = 0.0
        image[image > 1.0] = 1.0
        image *= 255
        image = image.astype(np.uint8)
        image = np.reshape(image, size)
        tifffile.imsave(filepath, image)    

    def predict_disparity(self):
        if self.settings.predict:
            if self.network is None:
                self.network = DenseMapNet(settings=self.settings)
                self.model = self.network.build_model()
            # gpu is slow in prediction during initial load of data
            # distorting the true speed of the network
            # we get the speed after 1 prediction
            if self.settings.images:
                self.get_epe(use_train_data=False, get_performance=True)
            else:
                for i in range(4):
                    self.get_epe(use_train_data=False, get_performance=True)
        else:
            # self.settings.images = True
            self.get_epe(use_train_data=False, get_performance=True)
            # self.get_epe(use_train_data=False)
            if self.settings.notrain:
                self.get_epe()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load checkpoint hdf5 file of model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        help=help_)
    parser.add_argument("-d",
                        "--dataset",
                        help="Name of dataset to load")
    parser.add_argument("-q",
                        "--pdir",
                        default="dataset",
                        help="Path for input data")
    parser.add_argument("-n",
                        "--num_dataset",
                        type=int,
                        help="Number of dataset file splits to load")
    help_ = "No training. Just prediction based on test data. Must load weights."
    parser.add_argument("-p",
                        "--predict",
                        action='store_true',
                        help=help_)
    help_ = "Generate images during prediction. Images are stored images/"
    parser.add_argument("-i",
                        "--images",
                        action='store_true',
                        help=help_)
    help_ = "No training. EPE benchmarking on test set. Must load weights."
    parser.add_argument("-t",
                        "--notrain",
                        action='store_true',
                        help=help_)
    help_ = "Best EPE"
    parser.add_argument("-e",
                        "--epe",
                        type=float,
                        help=help_)
    help_ = "No padding"
    parser.add_argument("-a",
                        "--nopadding",
                        action='store_true',
                        help=help_)
    help_ = "Mask images for sparse data"
    parser.add_argument("-m",
                        "--mask",
                        action='store_true',
                        help=help_)
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=2,
                        help="Number of samples in each mini-batch")	
    parser.add_argument("-r",
                        "--learning_rate",
                        type=float,
                        default=1e-2,
                        help="Learning rate")	
    parser.add_argument("-v",
                        "--no_data_value",
                        type=float,
                        default=-999.0,
                        help="No data value in disparity truth")	
    
    args = parser.parse_args()
    settings = Settings()
    settings.model_weights = args.weights
    settings.dataset = args.dataset
    settings.num_dataset = args.num_dataset
    settings.predict = args.predict
    settings.images = args.images
    settings.notrain = args.notrain
    settings.epe = args.epe
    settings.nopadding = args.nopadding
    settings.mask = args.mask
    settings.lr = args.learning_rate
    settings.batch_size = args.batch_size
    settings.no_data_value = args.no_data_value
    settings.pdir = args.pdir

	# TensorFlow allocates all GPU memory up front by default, so turn that off.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))
	
    # Run batch train, batch test, or predict.
    predictor = Predictor(settings=settings)
    if settings.predict:
        predictor.predict_disparity()
    else:
        predictor.train_network()
