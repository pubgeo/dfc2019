import argparse
import glob
import logging
import multiprocessing
import numpy as np
import os
import pickle
import pprint
import random
import sys

from pointset import PointSet


def get_list_of_files(folder):
    files = glob.glob(os.path.join(folder,"*_PC3.txt"))
    return sorted(files)


def load_file(pc3_path):
    logging.info("Loading '"+pc3_path+"'")
    
    cls_path = pc3_path[:-7]+'CLS.txt'
    
    pset = PointSet(pc3_path,cls_path)
    
    psets = pset.split()
    
    return [extract_data(ps) for ps in psets]


def extract_data(pset):
    data64 = np.stack([pset.x,pset.y,pset.z,pset.i,pset.r],axis=1)
    offsets = np.mean(data64,axis=0)
    variances = np.var(data64,axis=0)
    data = (data64-offsets).astype('float32')
    
    chist = {}
    for C in pset.c:
        if C in chist:
            chist[C]+=1
        else:
            chist[C]=1
    
    metadata = {
            "offsets":offsets,
            "variance":variances,
            "fname":pset.filename,
            "N":data.shape[0],
            "class_count":chist}
    
    return [metadata, data, pset.c]


def reduce_metadata(all_metadata):
    sum_N = 0
    cls_hist = {}
    sum_variances = np.zeros((5,))
    for d in all_metadata:
        sum_N += d['N']
        sum_variances += d['N']*d['variance']
        for cls,cnt in d['class_count'].items():
            if cls in cls_hist:
                cls_hist[cls] += cnt
            else:
                cls_hist[cls] = cnt
    
    compressed_label_map = {}
    decompress_label_map = {}
    
    i = 0
    for key in sorted(cls_hist):
        compressed_label_map[key] = i
        decompress_label_map[i] = key
        i+=1
    
    metadata = {
            "variance":sum_variances/sum_N,
            "cls_hist":cls_hist,
            "compressed_label_map":compressed_label_map,
            "decompress_label_map":decompress_label_map}
    return metadata

def parse_args(argv):
    # Setup arguments & parse
    parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i','--input-path', help='e.g. /path/to/DFC/Track4', required=True)
    parser.add_argument('-o','--output-path', help='e.g. /path/to/training_data_folder', required=True)
    parser.add_argument('-f','--training-frac', help='Fraction of data to use for training vs validation', default=0.9, type=float)
    
    return parser.parse_args(argv[1:])


def start_log(opts):
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    rootLogger = logging.getLogger()

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-3.3s] %(message)s")
    fileHandler = logging.FileHandler(os.path.join(opts.output_path,os.path.splitext(os.path.basename(__file__))[0]+'.log'),mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    logFormatter = logging.Formatter("[%(levelname)-3.3s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.level=logging.DEBUG

    logging.debug('Options:\n'+pprint.pformat(opts.__dict__))


def main(argv=None):
    opts = parse_args(argv if argv is not None else sys.argv)
    start_log(opts)
    
    files = get_list_of_files(opts.input_path)
    random.seed(0)
    random.shuffle(files)
    
    # Determine split
    ix = int(len(files)*opts.training_frac)
    
    plan = {'train':files[:ix],'val':files[ix:]}
    
    for key, files in plan.items():
        logging.info('Loading {} files for {}'.format(len(files),key))
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            data = pool.map(load_file, files)
        
        all_metadata = [datum[0] for filedata in data for datum in filedata]
        dataset = [datum[1] for filedata in data for datum in filedata]
        labels = [datum[2] for filedata in data for datum in filedata]
        
        metadata = reduce_metadata(all_metadata)
        
        with open(os.path.join(opts.output_path,"dfc_"+key+"_metadata.pickle"),'wb') as f:
            pickle.dump(metadata,f,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(opts.output_path,"dfc_"+key+"_dataset.pickle"),'wb') as f:
            pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(opts.output_path,"dfc_"+key+"_labels.pickle"),'wb') as f:
            pickle.dump(labels,f,pickle.HIGHEST_PROTOCOL)

    logging.info('Done')


if __name__ == '__main__':
    main(sys.argv)