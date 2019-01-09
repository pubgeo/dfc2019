__author__ = 'jhuapl'
__version__ = 0.1

import sys,os
import params
from dfcBaseline import DFCBaseline
from dataFunctions import parse_args

def main(argv):
    isTrain,mode = parse_args(argv, params)
    os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
    model = DFCBaseline(params=params, mode=mode)
    if isTrain:
        model.train()
    else:
        model.test()

if __name__=="__main__":
    main(sys.argv)



