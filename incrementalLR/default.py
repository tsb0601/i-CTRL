# Modified based on the MDEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.SEED = 10

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True
_C.CUDNN.NUM_WORKERS = 2

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.IMAGE_SIZE = [32, 32]
_C.MODEL.BACKBONE = 'dcgan'
_C.MODEL.PCACOMP = 15
_C.MODEL.SAMPLE_N = 10
_C.MODEL.SAMPLE_K = 16
_C.MODEL.NC = 3    # Number of channels in the training images. For color images this is 3
_C.MODEL.NZ = 128  # Size of z latent vector (i.e. size of generator input)
_C.MODEL.NGF = 64  # Size of feature maps in generator
_C.MODEL.NDF = 64  # Size of feature maps in discriminator

_C.TRAIN = CN()
_C.TRAIN.DATASET = 'cifar10'
_C.TRAIN.NUM_CLASSES = 10
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.BATCH_SIZE_PER_GPU = 128
_C.TRAIN.LAMBD = 15
_C.TRAIN.N_ITER_DIS = 0
_C.TRAIN.N_ITER_GEN = 0
_C.TRAIN.LRD = 0.0002
_C.TRAIN.LRG = 0.0002
_C.TRAIN.CLASS_PER_STEP = 1

_C.TRAIN.BETA1 = 0.5  # Beta1 hyperparam for Adam optimizers
_C.TRAIN.BETA2 = 0.999
_C.TRAIN.EPOCHS = 30
_C.TRAIN.MODE = 1
_C.TRAIN.CLASSI = 10
_C.TRAIN.CLASSII = 10
_C.TRAIN.EPS = 0.5
_C.TRAIN.START_CLASS = 0
_C.TRAIN.NETD_CKPT = ''
_C.TRAIN.NETG_CKPT = ''
_C.TRAIN.REVIEW = False
_C.TRAIN.REVIEW_ROUND = 12


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
