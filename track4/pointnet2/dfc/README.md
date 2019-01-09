# IEEE DFC 2019 Track 4 Baseline Algorithm

This model and code is the baseline algorithm for track 4 of the IEEE Data Fusion Contest (DFC) 2019.

In addition to pointnet2 (see [README](../README.md) in parent folder for details), this algorithm also relies heavily on [PDAL (Point Data Abstraction Library)](https://pdal.io).

## Docker Container

A docker container implementation has been provided for easy setup.  The container is based on a tensorflow-gpu image; therefore, nvidia-docker (version 2) must be installed on the host machine.  See https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0) for installation instructions.  Unfortunately, nvidia-docker does not support Windows; the dockerfile can likely be modified to use a non-gpu tensorflow image (which would allow the use of normal docker), but this has not been verified.

The docker container image can be built by running:
```
cd /path/to/pointnet2
docker build . -t dfc_pointcloud
```

The following steps assume the container name is `dfc_pointcloud`

## Creating Training Dataset
To reduce file IO costs at training time, the training data set should be precompiled:
1. Download the track 4 training dataset and extract it to `/path/to/pointnet2/data/dfc/train/Track4`.
1. Run the following:
```
docker run -it --rm \
    -v /path/to/pointnet2/data:/pointnet2/data \
    -v /path/to/pointnet2/dfc:/pointnet2/dfc \
    dfc_pointcloud python dfc/create_train_dataset.py \
    -i /pointnet2/data/dfc/train/Track4 -o /pointnet2/data/dfc/train
```
1. This will produce the following files:
  * `dfc_train_dataset.pickle`: Contains point clouds for the training dataset stored in numpy format
  * `dfc_train_labels.pickle`: Contains the per-point labels for each of the point clouds
  * `dfc_train_metadata.pickle`: Contains information related to scaling the different dimensions of the point clouds, as well as mapping the labels to/from a condensed sequential list
  * `dfc_val_*.pickle`: Same as above, but for a validation subset of the data (used for evaluating hyperparameters)

## Training Model
To train a model that includes the intensity and return number dimensions (3 & 4 respectively), run:
```
docker run --runtime=nvidia -it --rm \
    -v /path/to/pointnet2/data:/pointnet2/data \
    -v /path/to/pointnet2/dfc:/pointnet2/dfc \
    -v /path/to/pointnet2/log:/pointnet2/log \
    -v /path/to/pointnet2/models:/pointnet2/models \
    -v /path/to/pointnet2/utils:/pointnet2/utils \
    dfc_pointcloud python /pointnet2/dfc/train.py \
    --data_dir=data/dfc/train \
    --log_dir=data/dfc/model \
    --model=pointnet2_sem_seg_two_feat --extra-dims 3 4
```

## Using Pretrained Model
If using a pretrained model and not training from scratch, copy the model checkpoint files and associated `dfc_train_metadata.pickle` file to `/path/to/pointnet2/data/dfc/model`

## Running Inference
To run inference (i.e. to classify point clouds), do the following:
1. Copy the point cloud files (the *PC3.txt files) to `/path/to/pointnet2/data/dfc/inference_data/in`
1. Replacing the model.ckpt-###### with the appropriate name/number for your model, run the following code:
```
docker run --runtime=nvidia -it --rm \
    -v /path/to/pointnet2/data:/pointnet2/data \
    -v /path/to/pointnet2/dfc:/pointnet2/dfc \
    -v /path/to/pointnet2/log:/pointnet2/log \
    -v /path/to/pointnet2/models:/pointnet2/models \
    -v /path/to/pointnet2/utils:/pointnet2/utils \
    dfc_pointcloud python /pointnet2/dfc/inference.py \
    --model=pointnet2_sem_seg_two_feat --extra-dims 3 4 \
    --model_path=data/dfc/model/model.ckpt-###### \
    --input_path=data/dfc/inference_data/in \
    --output_path=data/dfc/inference_data/out
```
Note on previous step; if you wish to output both .las and CLS.txt files, add the following option to the command: `--output_type=BOTH`
