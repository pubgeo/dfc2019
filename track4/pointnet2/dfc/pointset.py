import argparse
import glob
import json
import logging
import multiprocessing
import numpy as np
import os
import pprint
import random
import subprocess
import sys
import tempfile

from laspy.file import File as LasFile
from laspy.header import Header as LasHeader
from pathlib import Path

# Point set (point cloud) class definition
class PointSet(object):
    def __init__(self, points_file, class_file=''):
        # Load points file
        if points_file.endswith('.las') or points_file.endswith('.laz'):
            lfile = LasFile(points_file, mode='r')
            self.x = np.copy(lfile.X).astype('f8')*lfile.header.scale[0]
            self.y = np.copy(lfile.Y).astype('f8')*lfile.header.scale[1]
            self.z = np.copy(lfile.Z).astype('f8')*lfile.header.scale[2]
            self.i = np.copy(lfile.Intensity).astype('f8')
            self.r = np.copy(lfile.return_num).astype('f8')
            self.c = np.copy(lfile.Classification)
            lfile.close()
        elif points_file.endswith('.txt'):
            data = np.loadtxt(points_file,delimiter=',',dtype='f8')
            self.x = data[:,0]
            self.y = data[:,1]
            self.z = data[:,2]
            self.i = data[:,3]
            self.r = data[:,4]
            if not class_file:
                if data.shape[1] > 5:
                    self.c = data[:,5].astype('uint8')
                else:
                    self.c = np.zeros(self.x.shape,dtype='uint8')
            else:
                self.c = np.loadtxt(class_file,dtype='uint8')
        else:
            raise ValueError('Unknown file type extension: '+points_file)
        self.filepath = points_file
        self.filename = os.path.splitext(os.path.basename(points_file))[0]
        if self.filename.endswith('_PC3'):
            self.filename = self.filename[:-4]

    def save(self, output_file, class_file=''):
        if output_file.endswith('.txt'):
            if (not class_file and self.c.any()):
                np.savetxt(output_file,
                        np.stack([self.x,self.y,self.z,self.i,self.r,self.c],axis=1),
                        fmt='%.2f,%.2f,%.2f,%d,%d,%d')
            else:
                np.savetxt(output_file,
                        np.stack([self.x,self.y,self.z,self.i,self.r],axis=1),
                        fmt='%.2f,%.2f,%.2f,%d,%d')
            if class_file:
                self.save_classifications_txt(class_file)
        elif output_file.endswith('.las') or output_file.endswith('.laz'):
            lfile = LasFile(output_file, mode='w', header=LasHeader(x_scale=0.01,y_scale=0.01,z_scale=0.01))
            lfile.X = self.x/0.01
            lfile.Y = self.y/0.01
            lfile.Z = self.z/0.01
            lfile.Intensity = self.i
            lfile.flag_byte = self.r
            lfile.Classification = self.c
            lfile.close()
        else:
            raise ValueError('Unknown file type extension: '+output_file)
    
    def save_canonical_txt(self, output_file_name_base):
        self.save(output_file_name_base+'_PC3.txt',
                class_file=output_file_name_base+'_CLS.txt')
                
    def save_classifications_txt(self, class_file):
        np.savetxt(class_file,self.c,fmt='%d')
    
    def split(self, points_per_chip=65536, overlap=True, pad=64):
        with tempfile.TemporaryDirectory() as tmpdir:
            ipath = os.path.join(tmpdir,self.filename+'.las')
            self.save(ipath)
            
            if overlap:
                charkey = [['A','B'],['C','D']]
                
                # get min/max x/y
                xmin = np.min(self.x)
                xmax = np.max(self.x)
                ymin = np.min(self.y)
                ymax = np.max(self.y)
                
                for i in range(2):
                    xcrop = [xmin+pad,xmax-pad] if i else [xmin-1,xmax+1]
                    for j in range(2):
                        ycrop = [ymin+pad,ymax-pad] if j else [ymin-1,ymax+1]
                        
                        opath = os.path.join(tmpdir,
                                self.filename+'_'+charkey[j][i]+'#.las')
                        
                        crop_bounds = '([{},{}],[{},{}])'.format(xcrop[0], xcrop[1], *ycrop)
                        
                        # Format pipeline string
                        if i==0 and j==0:
                            pipeline = {'pipeline':[
                                    ipath,
                                    {'type':'filters.voxelcentroidnearestneighbor'},
                                    {'type':'filters.chipper','capacity':'65536'},
                                    opath]}
                        else:
                            pipeline = {'pipeline':[
                                    ipath,
                                    {'type':'filters.crop','bounds':crop_bounds},
                                    {'type':'filters.voxelcentroidnearestneighbor'},
                                    {'type':'filters.chipper','capacity':str(points_per_chip)},
                                    opath]}
                        
                        p = subprocess.run(['/opt/conda/envs/cpdal-run/bin/pdal','pipeline','-s'],input=json.dumps(pipeline).encode())
                        if p.returncode:
                            raise ValueError('Failed to run pipeline: \n"'+json.dumps(pipeline)+'"')
                        
            else:
                opath = os.path.join(tmpdir,self.filename+'_#.las')
                pipeline = {'pipeline':[
                        ipath,
                        {'type':'filters.voxelcentroidnearestneighbor'},
                        {'type':'filters.chipper','capacity':'65536'},
                        opath]}
                p = subprocess.run(['/opt/conda/envs/cpdal-run/bin/pdal','pipeline','-s'],input=json.dumps(pipeline).encode())
                if p.returncode:
                    raise ValueError('Failed to run pipeline: \n"'+json.dumps(pipeline)+'"')
                
            return [PointSet(str(f)) for f in Path(tmpdir).glob(self.filename+'_*.las')]


if __name__ == '__main__':
    #main(sys.argv)
    pset = PointSet('/data/dfc_v4c/test2/classed/OMA/OMA_Tile_095_classes.las')