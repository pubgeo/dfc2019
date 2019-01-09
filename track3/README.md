# Track 3: Multi-View Semantic Stereo
The goal is to predict semantic labels and a digital surface model given several multi-view unrectified images associated with a pre-computed geometry model to focus on the data fusion problem and not on registration. Example python code is provided in the baseline solution to demonstrate epipolar rectification, triangulation, and coordinate conversion for the satellite images.

## Baseline Algorithm
For track 3, we used the ICNet and DenseMapNet models that we trained for [track 2](../track2). The MVS code provided demonstrates use of the RPC00B sensor model, UTM coordinate conversion, satellite image IMD image metadata, epipolar rectification, and triangulation. We also include an option to run SGBM for stereo which currently produces better results than DenseMapNet for MVS. To use this, set USE_SGM = True in [test-mvs.py](mvs/test-mvs.py).

These instructions assume you have configured an [Anaconda](https://www.anaconda.com/download/) environment with [dev-gpu.yml](../dev-gpu.yml) or [dev-cpu.yml](../dev-cpu.yml).

```bash
python ./mvs/test-mvs.py
```

Review the default file names for the inputs and outputs in [test-mvs.py](mvs/test-mvs.py) and ensure that they match the data downloaded for the contest.

Now the submission folder should have _CLS and _DSM files suitable for metric evaluation. Since you don't have the reference _CLS and _DSM files for the validation set used for leaderboard testing, you may try running MVS on a subset of the training data and then evaluate the leaderboard metrics using the truth _CLS and _DSM files provided for that data.
