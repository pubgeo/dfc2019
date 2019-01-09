__author__ = 'jhuapl'
__version__ = 0.1

import numpy as np

from segmentation_models import UnetRegressor,Unet
from dataFunctions import *

from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import load_model
from keras import backend as K

from tqdm import tqdm
import json
import tensorflow as tf

class DFCBaseline:
    def __init__(self, params=None, mode=None):
        """
        Initializes baseline by setting params, mode, and loading mean values to be used in the case non-RGB imagery is being used
        :param params: input parameters from params.py
        :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
        :return: None
        """
        self.params=params
        self.mode = mode
        self.meanVals = None
        if params.NUM_CHANNELS != 3:
            if params.MEAN_VALS_FILE is None:
                self.meanVals = np.zeros(params.NUM_CHANNELS).tolist()
            else:
                self.meanVals = json.load(open(params.MEAN_VALS_FILE))

        # TensorFlow allocates all GPU memory up front by default, so turn that off
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

    def test(self):
        """
        Launches testing, which saves output files
        :return: None
        """
        input_tensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS))
        input_shape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS)
        model = self.get_model(None, input_tensor, input_shape)
        
        if self.mode == self.params.SEMANTIC_MODE:
            numPredChannels = self.params.NUM_CATEGORIES
            outReplaceStr = self.params.CLSPRED_FILE_STR
            model = self.build_model()
            model.load_weights(self.params.SEMANTIC_TEST_MODEL, by_name=True)
        elif self.mode == self.params.SINGLEVIEW_MODE:
            numPredChannels = 1
            outReplaceStr = self.params.AGLPRED_FILE_STR
            model = self.build_model()
            model.load_weights(self.params.SINGLEVIEW_TEST_MODEL, by_name=True)
        
        model.summary()

        imgPaths = get_image_paths(self.params, isTest=True)
        print('Number of files = ', len(imgPaths))
        for imgPath in tqdm(imgPaths):
            imageName = os.path.split(imgPath)[-1]
            outName = imageName.replace(self.params.IMG_FILE_STR, outReplaceStr)
            img = np.expand_dims(load_img(imgPath), axis=0).astype('float32')
            img = image_batch_preprocess(img, self.params, self.meanVals)
            
            pred = model.predict(img)[0,:,:,:]
            
            if self.mode == self.params.SEMANTIC_MODE:
                if self.params.NUM_CATEGORIES > 1:
                    pred = np.argmax(pred, axis=2).astype('uint8')
                else:
                    pred = (pred > self.params.BINARY_CONF_TH).astype('uint8')
                if self.params.CONVERT_LABELS:
                    pred = convert_labels(pred, self.params, toLasStandard=True)
            else:
                pred = pred[:,:,0]
                
            tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred, compress=6)
            
    def train(self):
        """
        Launches training and stores checkpoints at a frequency defined within params
        :return: None
        """
        
        imgPaths = get_image_paths(self.params,isTest=False)
        if self.params.BLOCK_IMAGES:
            trainData = get_train_data(imgPaths, self.params)
        else:
            trainData = []
            for imgPath in imgPaths:
                trainData.append((imgPath,0,0))
            
        train_datagen = self.image_generator(trainData)
        model = self.build_model()
        model.summary()
        
        checkpoint = ModelCheckpoint(filepath=self.params.CHECKPOINT_PATH, monitor='loss', verbose=0, save_best_only=False,
                                         save_weights_only=False, mode='auto', period=self.params.MODEL_SAVE_PERIOD)
        
        # train model
        model.fit_generator(train_datagen, steps_per_epoch=int(len(trainData)/self.params.BATCH_SZ),
                            epochs=self.params.NUM_EPOCHS, callbacks=[checkpoint])
    
    
    def get_model(self, encoderWeights, input_tensor, input_shape):
        """
        Loads the model from segmentation_models for either semantic segmentation or single-view depth prediction
        :param encoderWeights: encoder weights for the backbone (e.g., imagenet)
        :param input_tensor: image shape for training
        :param input_shape: tensor shape for training
        :return model to be used for training/testing:
        """
        
        if self.mode == self.params.SINGLEVIEW_MODE:
            model = UnetRegressor(input_shape=input_shape, input_tensor=input_tensor, 
                        backbone_name=self.params.BACKBONE, encoder_weights=encoderWeights)
        elif self.mode == self.params.SEMANTIC_MODE:
            model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=self.params.BACKBONE, 
                         encoder_weights=encoderWeights, classes=self.params.NUM_CATEGORIES)
            
        return model
    
    def build_model(self):
        """
        Prepares either the single-view depth prediction model or the semantic segmentation model
            depending on params.py
        :return: model architecture to train
        """
               
        if self.params.BLOCK_IMAGES:
            imgSz = self.params.BLOCK_SZ
        else:
            imgSz = self.params.IMG_SZ
        
        input_tensor = Input(shape=(imgSz[0],imgSz[1],self.params.NUM_CHANNELS))
        input_shape = (imgSz[0],imgSz[1],self.params.NUM_CHANNELS)
        
        if self.params.CONTINUE_TRAINING:
            model = self.get_model(None, input_tensor, input_shape)
            if self.mode == self.params.SINGLEVIEW_MODE:
                print('Continuing trainining from %s' % self.params.CONTINUE_SINGLEVIEW_MODEL_FILE)
                model.load_weights(self.params.CONTINUE_SINGLEVIEW_MODEL_FILE)
            elif self.mode == self.params.SEMANTIC_MODE:
                print('Continuing trainining from %s' % self.params.CONTINUE_SEMANTIC_MODEL_FILE)
                model.load_weights(self.params.CONTINUE_SEMANTIC_MODEL_FILE)
        else:
            if self.params.NUM_CHANNELS > 3:
                input_tensor_rgb = Input(shape=(imgSz[0],imgSz[1],3))
                input_shape_rgb = (imgSz[0],imgSz[1],3)
                model = self.get_model(None, input_tensor, input_shape)
                if self.params.ENCODER_WEIGHTS is not None:
                    baseModel = self.get_model(self.params.ENCODER_WEIGHTS, input_tensor_rgb, input_shape_rgb)
                    print("Copying %s weights to %d-band model" % (self.params.ENCODER_WEIGHTS, self.params.NUM_CHANNELS))
                    for i in tqdm(range(len(baseModel.layers))):
                        if i>=7:
                            model.layers[i].set_weights(baseModel.layers[i].get_weights())
            else:
                model = self.get_model(self.params.ENCODER_WEIGHTS, input_tensor, input_shape)
            
            loss = self.params.SEMANTIC_LOSS
            if self.mode == self.params.SINGLEVIEW_MODE:
                loss=self.no_nan_mse

            model.compile(self.params.OPTIMIZER, loss=loss)
        
        return model
    
    def image_generator(self, trainData):
        """
        Generates training batches of image data and ground truth from either semantic or depth files
        :param trainData: training paths of CLS files (string replacement to get RGB and depth files) and starting x,y pixel positions (can be      non-zero if blocking is set to happen in params.py)
        :yield: current batch data
        """
        idx = np.random.permutation(len(trainData))
        while True:
            batchInds = get_batch_inds(idx, self.params)
            for inds in batchInds:
                imgBatch,labelBatch = load_batch(inds, trainData, self.params, self.mode, self.meanVals)
                yield (imgBatch, labelBatch)
    
    def no_nan_mse(self, y_true, y_pred):
        """
        Custom mean squared error loss function for single-view depth prediction used to ignore NaN/invalid depth values
        :param y_true: ground truth depth
        :param y_pred: predicted depth
        :return: mse loss without nan influence
        """
        mask_true = K.cast(K.not_equal(y_true, self.params.IGNORE_VALUE), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_mse

    