# Track 4: 3D Point Cloud Classification
Track 4 is the 3D Point Cloud Classification track.  The goal is to classify (semantically segment) point clouds on a per point basis.  The classes are:

| Class Index   | Class Description |
| :-----------: | ----------------- |
| 2             | Ground            |
| 5             | High Vegetation   |
| 6             | Building          |
| 9             | Water             |
| 17            | Bridge Deck       |

Additionally, some of the ground truth points are marked with a 0 class.  This represents unlabeled data, and points with this label will be ignored for metrics purposes.

## Baseline Algorithm
For the baseline algorithm, a PointNet++ (aka PointNet2) model was updated with modifications to support splitting/recombining large scenes.  For details on setting up/running the model, see [pointnet2/dfc/README.md](pointnet2/dfc/README.md)

## Metrics
To run the metrics code, it is easiest to use the same docker container that is used for the model, though it is not necessary.  Example command:

```
docker run -it --rm \
    -v /path/to/data:/data \
    -v /path/to/metrics_code_folder:/metrics \
    dfc_pointcloud bash -c \
    "python /metrics/track4-metrics.py -g /data/ground_truth -d /data/output_data | tee /data/output_data/metrics.txt"
```
