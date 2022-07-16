from __future__ import division, print_function

import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = "birds"
__C.CONFIG_NAME = ""
__C.VERSION = ""
__C.DATA_DIR = ""
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 6

__C.RNN_TYPE = "LSTM"  # 'GRU'
__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 5
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ""
__C.TRAIN.NET_G = ""
__C.TRAIN.B_NET_D = True

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0
__C.TRAIN.SMOOTH.ALPHA = 0.005
__C.TRAIN.SMOOTH.ALPHA_1 = 2


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18

# Path option
__C.PATH = edict()
__C.PATH.SAMPLES = ""


def update_config(args):
    # Dataset Name: birds
    __C.DATASET_NAME = args.dataset
    __C.VERSION = args.version
    __C.DATA_DIR = args.data_dir
    __C.GPU_ID = args.gpu_id
    if args.gpu_id >= 0:
        __C.CUDA = True
    else:
        __C.CUDA = False
    __C.WORKERS = args.num_workers
    __C.RNN_TYPE = args.rnn_type
    __C.EVAL = args.eval

    __C.TREE = edict()
    __C.TREE.BRANCH_NUM = args.num_branch
    __C.TREE.BASE_SIZE = args.base_size

    # Training options
    __C.TRAIN = edict()
    __C.TRAIN.BATCH_SIZE = args.batch_size
    __C.TRAIN.MAX_EPOCH = args.max_epoch
    __C.TRAIN.SNAPSHOT_INTERVAL = args.snapshot_interval
    __C.TRAIN.DISCRIMINATOR_LR = args.d_lr
    __C.TRAIN.GENERATOR_LR = args.g_lr
    __C.TRAIN.ENCODER_LR = args.encoder_lr
    __C.TRAIN.RNN_GRAD_CLIP = args.rnn_grad_clip
    __C.TRAIN.FLAG = args.train
    __C.TRAIN.NET_E = args.net_e
    __C.TRAIN.NET_G = args.net_g

    __C.TRAIN.SMOOTH = edict()
    __C.TRAIN.SMOOTH.GAMMA1 = args.smooth_gamma_1
    __C.TRAIN.SMOOTH.GAMMA2 = args.smooth_gamma_2
    __C.TRAIN.SMOOTH.GAMMA3 = args.smooth_gamma_3
    __C.TRAIN.SMOOTH.LAMBDA = args.smooth_lambda

    # Path option
    __C.PATH = edict()
    __C.PATH.SAMPLES = args.samples_dir

    # Model options
    __C.GAN = edict()
    __C.GAN.DF_DIM = args.df_dim
    __C.GAN.GF_DIM = args.gf_dim
    __C.GAN.Z_DIM = args.z_dim
    __C.GAN.CONDITION_DIM = args.condition_dim
    __C.GAN.R_NUM = args.num_residual

    __C.TEXT = edict()
    __C.TEXT.CAPTIONS_PER_IMAGE = args.caps_per_img
    __C.TEXT.EMBEDDING_DIM = args.text_emb_dim
    __C.TEXT.WORDS_NUM = args.words_num


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(("Type mismatch ({} vs. {}) " "for config key: {}").format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
