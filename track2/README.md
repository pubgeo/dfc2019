# Track 2: Pairwise Semantic Stereo
A pair of epipolar rectified images is given, and the objective is to predict semantic labels and stereo disparities (pixels).

## Baseline Algorithm
For track 2, we trained ICNet for semantic segmentation and DenseMapNet for stereo. The following instructions step through training, testing, and metric evaluation for both. These were run in Windows, but the code can just as easily run in Linux with minor edits.

These instructions assume you have configured an [Anaconda](https://www.anaconda.com/download/) environment with [dev-gpu.yml](../dev-gpu.yml) or [dev-cpu.yml](../dev-cpu.yml).


### TIFF Conversion
Convert TIFF files to NPZ files which are used to train the models. The image indices are shuffled and then training and validation data sets are produced. Change the input and output folders in the code before running. 

```bash
python make_track2_npz.py
```

### Training ICNet
Model weights are stored in the "checkpoints" folder, and a "tmp" folder is created to track mIoU scores for the validation set. When you're happy with a checkpoint, use that weight file for testing below.

```bash
python ./icnet/train.py ^
--train_name=../data/train/track2_npz/dfc2019.track2.train.left ^
--train_truth_name=../data/train/track2_npz/dfc2019.track2.train.left_label ^
--test_name=../data/train/track2_npz/dfc2019.track2.test.left ^
--test_truth_name=../data/train/track2_npz/dfc2019.track2.test.left_label ^
--batch_size=8 ^
--lr=1e-3 ^
--decay=1e-6 ^
--n_epochs=1000 ^
--n_trains=21
```
> Note: if using linux, convert the '^' characters above to '\' for multi line commands

### Training DenseMapNet
Update the folder names as needed (convert '^' to '\' if using linux). Model weights are stored in the "checkpoints" folder, and a "tmp" folder is created to track endpoint error for the validation set. When you're happy with a checkpoint, use that weight file for testing below. An example model file is also provided which was trained for less than one day.

```bash
python ./densemapnet/train.py ^
--pdir=../data/train/track2_npz/ ^
--dataset=dfc2019.track2 ^
--num_dataset=21 ^
-b=2 ^
-r=0.001 ^
--nopadding
```
> Note: if using linux, convert the '^' characters above to '\' for multi line commands

### Running ICNet and DenseMapNet on validation images
```bash
python test-icnet.py ../data/validate/Track2/ ../data/validate/Track2-Submission/ ./weights/190101-us3d.icnet.weights.18-3.h5
python test-densemapnet.py ../data/validate/Track2/ ../data/validate/Track2-Submission/ ./weights/181230-dfc2019.track2.densemapnet.weights.20-20.h5
```
SGBM currently produces better disparity estimates than DenseMapNet. While DenseMapNet currently produces far fewer gross outliers than SGBM, it does not capture fine details well.  

```bash
python test-sgbm.py ../data/validate/Track2/ ../data/validate/Track2-Submission/
```
