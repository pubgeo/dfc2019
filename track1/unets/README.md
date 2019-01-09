# Data Fusion Contest U-Net Baseline

U-Net baseline for image semantic segmentation and single-view depth prediction.

## Dependencies

The following library versions were used during training/testing:

* [albumentations](https://github.com/albu/albumentations) 0.1.2
* tifffile 0.15.1
* keras 2.2.4
* tqdm 4.26.0
* Modified version of [segmentation_models](https://github.com/qubvel/segmentation_models) to perform single-view depth estimation


## Training and Testing

For both semantic segmentation and single-view depth prediction, training parameters can be adjusted in 'params.py'.

Command line arguments are used to determine train/test mode and semantic/single-view depth prediction mode. 

* Semantic segmentation train: ```python runBaseline.py train semantic```
* Single-view depth prediction train: ```python runBaseline.py train single-view```
* Semantic segmentation test: ```python runBaseline.py test semantic```
* Single-view depth prediction test: ```python runBaseline.py test single-view```

