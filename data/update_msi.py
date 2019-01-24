import argparse
import glob
import os
from osgeo import gdal
import tifffile


# read TIF image using TIFFFILE and write again using GDAL
def update_msi(input_file_name, output_file_name):
    img = tifffile.imread(input_file_name)
    rows, cols, bands = img.shape
    driver = gdal.GetDriverByName("GTiff")
    output_data = driver.Create(output_file_name, rows, cols, bands, gdal.GDT_UInt16)
    for band in range(0, bands):
        output_band = output_data.GetRasterBand(band + 1)
        output_band.WriteArray(img[:, :, band])
    output_data.FlushCache()
    output_data = None


# main
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    # get list of input files
    print(args.input_folder)
    files = glob.glob(args.input_folder + '*MSI*.tif')
    number_files = len(files)
    print('Number of files = ', number_files)
    for i in range(number_files):
        basename = os.path.basename(files[i])
        output_name = args.output_folder + basename
        input_name = args.input_folder + basename
        update_msi(input_name, output_name)

# example:
# python update_msi.py U:/TRAIN/Track1-MSI/ U:/TRAIN/Track1-MSI-GDAL/

