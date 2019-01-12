# Minor modifications added by Myron Brown, 2018
# - Original code adapted from https://github.com/roatienza/densemapnet
# - Replaced Conv2D with SeparableConv2D to reduce number of model parameters
# - Removed sigmoid activation to directly regress disparity values
# - Added option to run dropout at test time
# Scott Almes performed pep fixes and

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

import keras
from keras.layers import Dropout
from keras.layers import Input, Conv2DTranspose, SeparableConv2D
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
import time


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 0:
            sec = "%0.2f" % (sec * 100)
            return sec + " msec"
        elif sec < 60:
            sec = "%0.4f" % sec
            return sec + " sec"
        elif sec < (60 * 60):
            sec = "%0.4f" % (sec / 60)
            return sec + " min"
        else:
            sec = "%0.4f" % (sec / (60 * 60))
            return sec + " hr"

    def elapsed_time(self):
        delta = time.time() - self.start_time
        return delta

    def print_elapsed_time(self):
        print("Speed: %s " % self.elapsed(self.elapsed_time()))


class Settings(object):
    def __init__(self):
        self.ydim = 1024
        self.xdim = 1024
        self.channels = 3
        self.model_weights = None
        self.dropout_test = False


class DenseMapNet(object):
    def __init__(self, settings):
        self.settings = settings
        self.xdim = self.settings.xdim
        self.ydim = self.settings.ydim
        self.channels = self.settings.channels
        self.dropout_test = self.settings.dropout_test
        self.model = None

    def build_model(self):
        dropout = 0.2

        shape = (None, self.ydim, self.xdim, self.channels)
        left = Input(batch_shape=shape)
        right = Input(batch_shape=shape)

        # left image as reference
        x = SeparableConv2D(filters=16, kernel_size=5, padding='same')(left)
        xleft = SeparableConv2D(filters=1,
                                kernel_size=5,
                                padding='same',
                                dilation_rate=2)(left)

        # left and right images for disparity estimation
        xin = keras.layers.concatenate([left, right])
        xin = SeparableConv2D(filters=32, kernel_size=5, padding='same')(xin)

        # image reduced by 8
        x8 = MaxPooling2D(8)(xin)
        x8 = BatchNormalization()(x8)
        x8 = Activation('relu', name='downsampled_stereo')(x8)

        num_dilations = 4
        dilation_rate = 1
        y = x8
        # correspondence network
        # parallel cnn at increasing dilation rate
        for i in range(num_dilations):
            a = SeparableConv2D(filters=32,
                                kernel_size=5,
                                padding='same',
                                dilation_rate=dilation_rate)(x8)
            a = Dropout(dropout)(a, training=self.dropout_test)
            y = keras.layers.concatenate([a, y])
            dilation_rate += 1

        dilation_rate = 1
        x = MaxPooling2D(8)(x)
        # disparity network
        # dense interconnection inspired by DenseNet
        for i in range(num_dilations):
            x = keras.layers.concatenate([x, y])
            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = SeparableConv2D(filters=64,
                                kernel_size=1,
                                padding='same')(y)

            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = SeparableConv2D(filters=16,
                                kernel_size=5,
                                padding='same',
                                dilation_rate=dilation_rate)(y)
            y = Dropout(dropout)(y, training=self.dropout_test)
            dilation_rate += 1

        # disparity estimate scaled back to original image size
        x = keras.layers.concatenate([x, y], name='upsampled_disparity')
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters=32, kernel_size=1, padding='same')(x)
        x = UpSampling2D(8)(x)
        if not self.settings.nopadding:
            x = ZeroPadding2D(padding=(2, 0))(x)

        # left image skip connection to disparity estimate
        x = keras.layers.concatenate([x, xleft])
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = SeparableConv2D(filters=16, kernel_size=5, padding='same')(y)

        x = keras.layers.concatenate([x, y])
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2DTranspose(filters=1, kernel_size=9, padding='same')(y)

        # densemapnet model
        self.model = Model([left, right], y)

        if self.settings.model_weights:
            print("Loading checkpoint model weights %s...."
                  % self.settings.model_weights)
            self.model.load_weights(self.settings.model_weights)

        #        print("DenseMapNet Model:")
        #        self.model.summary()
        #        plot_model(self.model, to_file='densemapnet.png', show_shapes=True)

        return self.model
