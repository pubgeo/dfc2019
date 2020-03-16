# IEEE Data Fusion Contest
For more information visit IEEE's [2019 Data Fusion Contest](http://www.grss-ieee.org/community/technical-committees/data-fusion/2019-ieee-grss-data-fusion-contest/). Additionally, this repo contains the (poster)[DFC2019-WACV2019-Poster.pdf] presented at WACV 2019.

## Cloning this repository
This repository contains large model files used by baseline algorithms. The model files do not need to be downloaded, but are provided as a convenience. Please use [git lfs](https://git-lfs.github.com/) when cloning to have access to these models. If you did not install and initialize git lfs before cloning, you can simply run `git lfs fetch` after locally initializing git lfs.

## Contest Tracks
### Track 1: Single View Semantic 3D
In [Track 1](track1) an unrectified single-view image is provided for each geographic tile. The objective is to predict semantic labels and above-ground heights (meters).

### Track 2: Pairwise Semantic Stereo
In [Track 2](track2) a pair of epipolar rectified images is given, and the objective is to predict semantic labels and stereo disparities (pixels).

### Track 3: Multi-View Semantic Stereo
In [Track 3](track3) the goal is to predict semantic labels and a digital surface model given several multi-view unrectified images associated with a pre-computed geometry model to focus on the data fusion problem and not on registration. Example python code is provided in the baseline solution to demonstrate epipolar rectification, triangulation, and coordinate conversion for the satellite images.

### Track 4: 3D Point Cloud Classification
In [Track 4](track4) the aim is to label points from the given aerial point cloud according to several predetermined semantic classes. For this track only, performance is assessed using standard mIoU.

## Performance
For tracks 1-3, performance is assessed using the pixel-wise mean Intersection over Union (mIoU) for which true positives must have both the correct semantic label and height error less than a given threshold (1 meter for heights or 3 pixels for disparities). We call this metric mIoU-3.

## Data
JHU/APL has provided a large data set, including ground truth, for training and testing. Instructions for acquiring and using the data are located in the [data](data) directory.

## Baseline algorithms:
JHU/APL has developed baseline implementations in python for each challenge track to demonstrate how to manipulate the challenge data and produce valid submission files. These baselines are available in the Track 1-4 folders referenced above.

## Submission requirements:
Submissions must match the reference file formats and data types and must be readable by `scipy.misc.imread`. Please check your files before submitting. There is no requirement to use a particular language for producing submissions.

## Acknowledgements
The authors are grateful to the IEEE GRSS IADF committee chairs – Bertrand Le Saux, Ronny Hänsch, and Naoto Yokoya – for their collaboration in leveraging this work to enable public research. Commercial satellite images were provided courtesy of DigitalGlobe. U. S. Cities LiDAR and vector data were made publicly available by the Homeland Security Infrastructure Program. Geomni LiDAR and oblique imagery will be made available publicly for single use research purposes. This work was supported by the Intelligence Advanced Research Projects Activity (IARPA) contract no. 2017-17032700004. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA or the U.S. Government.

## Reference
For more information on the data:
[Marc Bosch, Kevin Foster, Gordon Christie, Sean Wang, Gregory D. Hager, and Myron Brown, "Semantic Stereo for Incidental Satellite Images," Proc. of Winter Conf. on Applications of Computer Vision, 2019.](https://arxiv.org/abs/1811.08739)

