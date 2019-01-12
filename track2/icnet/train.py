import cv2
import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import time
import os
from scipy import misc

from model_icnet import build_icnet
import memory_saving_gradients


# lazyleaf response on https://github.com/keras-team/keras/issues/9395
# Ref: salehi17, "Tversky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard or IoU)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
# modified by M. Brown for 2D images and to ignore UNLABELED category
def tversky_loss(y_true, y_pred):
    # ignore the last category
    shp = K.shape(y_true)
    y_true = y_true[:, :, :, 0:shp[3] - 1]
    y_pred = y_pred[:, :, :, 0:shp[3] - 1]

    alpha = 1.0
    beta = 1.0
    ones = K.ones(K.shape(y_true))
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true
    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))
    T = K.sum(num / den)
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


class us3d_trainer(object):

    # initialize and load data
    def __init__(self, args):
        self.args = args
        self.best_mIoU = 0.0
        print('Loading data...')
        self.load_test_data()
        print('Finished loading data')
        self.model = None
        np.random.seed(0)

    # load test data set
    def load_test_data(self):
        self.test_images = np.load(self.args.test_name + '.npz')['arr_0']
        self.test_images = (self.test_images - 127.5) / 255.0
        self.test_truth = np.load(self.args.test_truth_name + '.npz')['arr_0']

    # load train data set
    def load_train_data(self, num=1):
        print('Loading ', self.args.train_name + '.%d.npz' % num)
        self.train_images = np.load(self.args.train_name + '.%d.npz' % num)['arr_0']
        self.train_images = (self.train_images - 127.5) / 255.0
        self.train_truth = np.load(self.args.train_truth_name + '.%d.npz' % num)['arr_0']
        self.n_classes = np.max(self.train_truth) + 1

        # create downsampled truth images for cascade guidance
        num = len(self.train_truth)
        height = self.train_truth[0].shape[0]
        width = self.train_truth[0].shape[1]
        self.Y0 = np.zeros((num, height, width, self.n_classes), dtype='float32')
        self.Y1 = np.zeros((num, height // 4, width // 4, self.n_classes), dtype='float32')
        self.Y2 = np.zeros((num, height // 8, width // 8, self.n_classes), dtype='float32')
        self.Y3 = np.zeros((num, height // 16, width // 16, self.n_classes), dtype='float32')
        for i in range(num):
            self.Y0[i] = to_categorical(
                cv2.resize(self.train_truth[i, :, :], (height, width), interpolation=cv2.INTER_NEAREST).astype(int),
                self.n_classes).reshape((height, width, -1))
            self.Y1[i] = to_categorical(cv2.resize(self.train_truth[i, :, :], (height // 4, width // 4),
                                                   interpolation=cv2.INTER_NEAREST).astype(int),
                                        self.n_classes).reshape((height // 4, width // 4, -1))
            self.Y2[i] = to_categorical(cv2.resize(self.train_truth[i, :, :], (height // 8, width // 8),
                                                   interpolation=cv2.INTER_NEAREST).astype(int),
                                        self.n_classes).reshape((height // 8, width // 8, -1))
            self.Y3[i] = to_categorical(cv2.resize(self.train_truth[i, :, :], (height // 16, width // 16),
                                                   interpolation=cv2.INTER_NEAREST).astype(int),
                                        self.n_classes).reshape((height // 16, width // 16, -1))

            # train the model

    def train_model(self):

        # setup checkpoint folder
        checkdir = "checkpoint"
        try:
            os.mkdir(checkdir)
        except FileExistsError:
            print("Folder exists: ", checkdir)

        # loop on epochs
        lr = self.args.lr + self.args.decay
        for i in range(1, self.args.n_epochs + 1):

            # update the learning rate
            lr = lr - self.args.decay

            # randomize order of train files
            indexes = np.arange(1, self.args.n_trains + 1)
            np.random.shuffle(indexes)

            # loop on train files
            is_compiled = False
            for j in indexes:

                # load training file
                self.load_train_data(j)

                # setup callback for writing new model checkpoint files
                filename = "us3d.icnet.weights.%d-%d.h5" % (i, j)
                filepath = os.path.join(checkdir, filename)
                checkpoint = ModelCheckpoint(filepath=filepath, save_weights_only=True, verbose=1, save_best_only=False)
                predict_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.compute_accuracy())
                callbacks = [checkpoint, predict_callback]

                # build the model, compile, and fit
                height = self.train_images[0].shape[0]
                width = self.train_images[0].shape[1]
                bands = self.train_images[0].shape[2]
                myloss = tversky_loss
                if self.model is None:
                    self.model = build_icnet(height, width, bands, self.n_classes, weights_path=self.args.checkpoint,
                                             train=True)
                if not is_compiled:
                    self.model.compile(optimizer=Adam(lr=lr), loss=myloss, loss_weights=[1.0, 0.4, 0.16])
                    is_compiled = True
                self.model.fit(self.train_images, [self.Y1, self.Y2, self.Y3], epochs=1,
                               batch_size=self.args.batch_size, shuffle=True, callbacks=callbacks)

    # convert category value image to RGB color image
    def category_to_color(self, category_image):

        # define colors
        # color table is here: https://www.rapidtables.com/web/color/RGB_Color.html
        colors = []
        colors.append((165, 42, 42))  # 0  brown (ground)
        colors.append((0, 128, 0))  # 1  green (trees)
        colors.append((255, 0, 0))  # 2  red   (buildings)
        colors.append((0, 0, 255))  # 3  blue  (water)
        colors.append((128, 128, 128))  # 4  gray  (elevated road / bridge)
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

    # save image with truth and prediction
    def save_image(self, num, rgb_image, truth_image, prediction_image):
        rows = truth_image.shape[0]
        cols = truth_image.shape[1]
        merged_image = np.zeros((rows, cols * 3, 3))

        rgb = rgb_image
        rgb = rgb.astype(np.uint8)
        rgb = np.reshape(rgb, [rows, cols, 3])

        truth = self.category_to_color(truth_image)
        prediction = self.category_to_color(prediction_image)

        merged_image[0:rows, 0:cols, :] = rgb
        merged_image[0:rows, cols:2 * cols, :] = truth
        merged_image[0:rows, 2 * cols:3 * cols, :] = prediction

        filename = 'test_' + '{:03d}'.format(num) + '.png'
        filepath = os.path.join("images", filename)
        misc.imsave(filepath, merged_image)

    # compute train and test accuracy
    def compute_accuracy(self):
        # this is slow, so only do it roughly every tenth input npz file at random
        num = np.random.random_sample()
        if num < 0.1:
            print('ACCURACY FOR VALIDATION SAMPLES:')
            self.predict_and_report_accuracy(self.test_images, self.test_truth)

    # update stats for mIoU computation
    def update_stats(self, seg, gt, tp, fp, fn, num_categories):
        for cat in range(num_categories):
            tp[cat] += ((seg == cat) & (gt == cat) & (gt < num_categories)).sum()
            fp[cat] += ((seg == cat) & (gt != cat) & (gt < num_categories)).sum()
            fn[cat] += ((seg != cat) & (gt == cat) & (gt < num_categories)).sum()

    # test the model for semantic segmentation accuracy
    def predict_and_report_accuracy(self, rgb_images, truth_images):

        # run prediction on all images and compute accuracy
        height = rgb_images[0].shape[0]
        width = rgb_images[0].shape[1]
        num = len(rgb_images)
        elapsed_total = 0.0
        max_category = self.n_classes - 1
        tps = np.zeros((max_category))
        tns = np.zeros((max_category))
        fps = np.zeros((max_category))
        fns = np.zeros((max_category))
        for i in range(num):
            start_time = time.time()
            indexes = np.arange(i, i + 1)
            rgb = rgb_images[indexes, :, :, :]
            prediction = self.model.predict(rgb)
            prediction = prediction[0]
            if prediction.shape[0] == 1:
                prediction = prediction.reshape(prediction.shape[1], prediction.shape[2], prediction.shape[3])
            elapsed_total += (time.time() - start_time)

            # loop on all categories and get labels from categorical scores
            # update this after accounting explicitly for an unknown category
            truth = truth_images[i, :, :]
            labels = np.zeros((height, width))

            for i in range(max_category):
                pred = np.argmax(prediction[:, :, 0:max_category], axis=2)
                pred = cv2.resize(pred, (height, width), interpolation=cv2.INTER_NEAREST)
                labels = pred
                self.update_stats(labels, truth, tps, fps, fns, max_category)

            # write truth and prediction image
            if self.args.save_images:
                self.save_image(i, rgb[0] * 255, truth, labels)

        mIoU = 0.0
        for j in range(max_category):
            iou = tps[j] / (tps[j] + fps[j] + fns[j])
            print("  IoU: %0.4f" % iou)
            mIoU += iou
        mIoU /= max_category
        print("mIoU: %0.4f" % mIoU)
        print("Speed: %0.4f seconds" % (elapsed_total / num))
        print("Speed: %0.4f Hz" % (num / elapsed_total))
        print(" ")
        if mIoU > self.best_mIoU:
            self.best_mIoU = mIoU
            print('-------- BEST mIoU = ', mIoU, '--------')

            # write to temp file
            tmpdir = "tmp"
            try:
                os.mkdir(tmpdir)
            except FileExistsError:
                print("Folder exists: ", tmpdir)
            filename = open('tmp/miou.txt', 'a')
            datetime = time.strftime("%H:%M:%S")
            for j in range(max_category):
                iou = tps[j] / (tps[j] + fps[j] + fns[j])
                filename.write("  IoU: %0.4f\n" % iou)
            filename.write("%s : mIoU: %f : Speed (Hz): %f\n" % (datetime, mIoU, (num / elapsed_total)))
            filename.close()
        else:
            print('Best was mIoU = ', self.best_mIoU)


# main program to train and validate an ICNet
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_name', type=str, default=None, help='train image NPZ file')
    parser.add_argument('--test_name', type=str, default=None, help='test image NPZ file')
    parser.add_argument('--train_truth_name', type=str, default=None, help='train truth image NPZ file')
    parser.add_argument('--test_truth_name', type=str, default=None, help='test truth image NPZ file')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of the optimizer')
    parser.add_argument('--decay', type=float, default=1e-6, help='learning rate decay (per epoch)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint to resume training')
    parser.add_argument('--save_images', type=bool, default=False, help='set flag to write images')
    parser.add_argument('--n_trains', type=int, default=1, help='number of training files')
    args = parser.parse_args()
    print(args)

    # TensorFlow allocates all GPU memory up front by default, so turn that off
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))

    # Save memory using gradient checkpoints
    K.__dict__["gradients"] = memory_saving_gradients.gradients_speed

    # initialize and train the model
    us3d = us3d_trainer(args)
    us3d.train_model()
