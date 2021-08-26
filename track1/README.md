# Track 1: Single View Semantic 3D 

An unrectified single-view image is provided for each geographic tile. The objective is to predict semantic labels and above-ground heights (meters).

## Baseline Algorithm
For track 1, JHU/APL trained a U-Net for for image semantic segmentation and single-view depth prediction. These were run in Windows, but the code can just as easily run in Linux with minor edits to the [params.py](unets/params.py) file which specifies folder locations.

Instructions for training and testing image semantic segmentation and single-view depth prediction models are provided in [unets/README.md](unets/README.md)

### Model weights

The model weights are now in a GitHub release zip file for download to avoid having large files in the code repo. 

### Quick Run
The following instructions are for running the baseline solution and assumes you have configured an [Anaconda](https://www.anaconda.com/download/) environment with [dev-gpu.yml](../dev-gpu.yml) or [dev-cpu.yml](../dev-cpu.yml).

#### Data layout
Due do the necessary data packaging splits, to run the following algorithms you will need to point [params.py](unets/params.py)'s [TRAIN_DIR](https://github.com/pubgeo/dfc2019/blob/master/track1/unets/params.py#L9) to a folder containing the Track1 AGL, CLS, RGB and MSI tif files found in the Train-Track1-RGB, Train-Track1-MSI-*, and Train-Track1-Truth zip files.
Similarly [params.py](unets/params.py)'s [TEST_DIR](https://github.com/pubgeo/dfc2019/blob/master/track1/unets/params.py#L12) should point to a folder containing MSI and RGB files found in the Validate-Track1 zip files.

#### Running
To train the UNET single view model, modify [params.py](unets/params.py) with your file names and run: 
```bash
python ./unets/runBaseline.py train single-view
```

To train train the UNET semantic model, modify [params.py](unets/params.py) with your file names and run:
```bash
python ./unets/runBaseline.py train semantic
```

To test the semantic or single-view models, modify [params.py](unets/params.py) with your file names and run:
```bash
python ./unets/runBaseline.py test semantic
```
or
```bash
python ./unets/runBaseline.py test single-view
```

## Testing Solution
After running tests for the images in the validation set folder, the submission folder should have _CLS and _AGL files suitable for metric evaluation. Since you don't have the reference _CLS and _AGL files for the validation set used for leaderboard testing, you may try running the predictions above on hold-out images and then evaluate the leaderboard metrics on those files.
```bash
python .\metrics\track1.py [submission folder] [truth folder]
```

Here, the truth folder is a folder containing the holdouts, while the submission folder contains your predictions.
