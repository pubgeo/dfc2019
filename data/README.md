# DATA 
Torrent files for all data provided to participants of the contest are now available in this repository.

## Data provided:
Both RGB and 8-band Multi-Spectral Images (MSI) are provided for Tracks 1-3. All source images are from WorldView-3 and are provided courtesy DigitalGlobe. MSI images are much larger than RGB due to the increased number of bands. The MSI images have been pan-sharpened. Submissions for Tracks 1-3 may use either RGB or MSI images. Both will be provided for validation and test phases of the contest as inputs. Use of the satellite image metadata provided is demonstrated in the baseline Track 3 solution, and additional information is available here: https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/106/ISD_External.pdf. For track 3, RPC sensor model metadata is retained in the RGB image files with adjustments to account for registration and image cropping. Sample code for manipulating the RPC metadata is provided in the Track 3 baseline code.

| Zip File               | DESCRIPTION |
| ---------------------- | ----------- |
|Train-Track1-RGB        | Track 1 RGB images |
|Train-Track1-MSI-*      | Track 1 MSI images (provided separately because they are large) |
|Train-Track1-Truth      | Track 1 reference data (AGL, CLS) for training |
|Validate-Track1         | Track 1 RGB and MSI images for leaderboard validation - reference data not provided |
|Train-Track2-RGB-*      | Track 2 RGB images |
|Train-Track2-MSI-*      | Track 2 MSI images |
|Train-Track2-Truth      | Track 2 reference data (DSP, CLS) for training |
|Validate-Track2         | Track 2 RGB and MSI images for leaderboard validation - reference data not provided |
|Train-Track3-RGB-*      | Track 3 RGB images |
|Train-Track3-MSI-*      | Track 3 MSI images |
|Train-Track3-Truth      | Track 3 reference data (DSM, CLS, and TXT indicating geospatial coordinates for submission) |
|Track3-Metadata         | Satellite metadata for each source image useful for multi-view stereo in Track 3 |
|Validate-Track3         | Track 3 RGB and MSI images for leaderboard validation - reference data not provided |
|Validate-Track3-Bounds  | Track 3 TXT files indicating geospatial coordinates for submission |
|Train-Track4            | Track 4 point clouds (PC3) |
|Train-Track4-Truth		 | Track 4 reference data (CLS) |
|Validate-Track4         | Track 4 point clouds for leaderboard validation - reference data not provided |

>The data package is broken up into many zips due to file hosting limitations. Folders with a trailing '-\*' imply that they were split into multiple folders due to file hosting limitations. 
For your own convenience, you may want to merge these folders together, removing the trailing '-\*'

> The tifffile.imwrite() function was used to produce Track 1 and Track 2 8-band MSI files, and 
it writes the metadata such that it's incompatible with GDAL and other TIF image readers but
such that tifffile.imread() does read them correctly. For those who prefer not to use tifffile
in their software for challenge solutions, [update_msi.py](update_msi.py) reads a folder of MSI images using 
TIFFILE and then rewrites them to a new folder using GDAL.

## Classification labels
Classification labels for all tracks are based on the LAS specification. Submissions must label semantic categories using these values consistent with the reference data provided. Unlabeled points in reference data are not included in metric evaluation. Any unlabeled points in a submission will be penalized in metric evaluation.

| ID  | Classification         |
| --- | :--------------------- | 
| 02  | Ground                 |
| 05  | Trees                  |
| 06  | Buildings              |
| 09  | Water                  |
| 17  | Bridge / elevated road |
| 65  | Unlabeled              |

## File Formats
### Point-cloud file format:

| Ext  | TYPE                | DESCRIPTION |
| ---- | ------------------- | ----------- |
| PC3  | txt                 | X, Y, Z, intensity, return number |
| CLS  | txt                 | Classification label of the point at same row in the corresponding point cloud file. |
	
### Image file formats:

| Post fix | TYPE      | BANDS     | DESCRIPTION |
| -------- | --------- | --------- | ----------- |
| CLS      | uint8     | 1         | Classification label |
| AGL      | float32   | 1         | Above ground level height (meters) |
| DSP      | float32   | 1         | Ground truth left disparity (pixels) |
| DSM      | float32   | 1         | Ground truth WGS84 Z coordinate (meters) |
| RGB      | uint8     | 3         | RGB input image |
| MSI      | uint16    | 8         | 8-band Visible and Near Infrared (V-NIR) MSI input image |

Note: `imageio` and `scipy.misc` do not handle 8 band images, so use `gdal` or `tifffile` if using those files.

## File naming and numbering conventions:
Images for tracks 1 and 3 are named with a three letter code for geographic location (JAX for Jacksonville and OMA for Omaha) followed by the specific geographic tile number and then the source image number. For instance, JAX_163_010_RGB is an RGB image for Jacksonville tile 163 from satellite image 10. The satellite image numbers match the file names for the IMD and RPB metadata files in the satellite image metadata folder provided.

Images for track 2 include source satellite image numbers for both the left and right images in an epipolar rectified pair. For instance, JAX_163_010_006_LEFT_RGB and JAX_163_010_006_RIGHT_RGB are epipolar rectified RGB images for Jacksonville tile 163 from source images 10 and 6.

Point clouds for track 4 include the three letter code for geographic location followed by the specific geographic tile number. For instance, JAX_114_PC3 is a 3D point cloud for Jacksonville tile 114.

## Train Image sizes:

| Track   | Resolution (Units) | Image Type |
| ------- | ------------------ | ---------- |
| Track-1 | 1024x1024 (pixels) | Unrectified images |
| Track-2 | 1024x1024 (pixels) | Epipolar rectified images |
| Track-3 | 2048x2048 (pixels) | Unrectified images with padding to allow for epipolar rectification and cropping |
| Track-4 | 512x512 (meters)   | 3D point cloud |
