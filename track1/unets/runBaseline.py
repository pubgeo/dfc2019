__author__ = 'jhuapl'
__version__ = 0.1

import sys
import os
import params
from dfcBaseline import DFCBaseline
from dataFunctions import parse_args


def main(argv):
    training, mode = parse_args(argv, params)
    os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
    model = DFCBaseline(params=params, mode=mode)
    if training:
        try:
            model.train()
        except ValueError as error:
            print(error, file=sys.stderr)
    else:
        model.test()


if __name__ == "__main__":
    main(sys.argv)
