# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_centernet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    # centernet config
    _C.MODEL.CENTERNET = CN()
    _C.MODEL.CENTERNET.DECONV_CHANNEL = [512, 256, 128, 64]
    _C.MODEL.CENTERNET.DECONV_KERNEL = [4, 4, 4]
    _C.MODEL.CENTERNET.NUM_CLASSES = 80
    _C.MODEL.CENTERNET.MODULATE_DEFORM = True
    _C.MODEL.CENTERNET.USE_DEFORM = True
    _C.MODEL.CENTERNET.BIAS_VALUE = -2.19
    _C.MODEL.CENTERNET.DOWN_SCALE = 4
    _C.MODEL.CENTERNET.MIN_OVERLAP = 0.7
    _C.MODEL.CENTERNET.TENSOR_DIM = 128
    _C.MODEL.CENTERNET.OUTPUT_SIZE = [128, 128]
    _C.MODEL.CENTERNET.BOX_MINSIZE = 1e-5
    _C.MODEL.CENTERNET.RESIZE_TYPE = "ResizeShortestEdge"
    _C.MODEL.CENTERNET.TRAIN_PIPELINES = [
        # ("CenterAffine", dict(boarder=128, output_size=(512, 512), random_aug=True)),
        ("RandomFlip", dict()),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
    ]
    _C.MODEL.CENTERNET.TEST_PIPELINES = []
    _C.MODEL.CENTERNET.LOSS = CN()
    _C.MODEL.CENTERNET.LOSS.CLS_WEIGHT = 1
    _C.MODEL.CENTERNET.LOSS.WH_WEIGHT = 0.1
    _C.MODEL.CENTERNET.LOSS.REG_WEIGHT = 1
    _C.MODEL.CENTERNET.LOSS.NORM_WH = False
    _C.MODEL.CENTERNET.LOSS.SKIP_LOSS = False
    _C.MODEL.CENTERNET.LOSS.SKIP_WEIGHT = 1.0
    _C.MODEL.CENTERNET.LOSS.MSE = False
    _C.MODEL.CENTERNET.LOSS.IGNORE_UNLABEL = False

    _C.MODEL.CENTERNET.LOSS.COMMUNISM = CN()
    _C.MODEL.CENTERNET.LOSS.COMMUNISM.ENABLE = False
    _C.MODEL.CENTERNET.LOSS.COMMUNISM.CLS_LOSS = 1.5
    _C.MODEL.CENTERNET.LOSS.COMMUNISM.WH_LOSS = 0.3
    _C.MODEL.CENTERNET.LOSS.COMMUNISM.OFF_LOSS = 0.1

    _C.MODEL.CENTERNET.IMGAUG_PROB = 2.0

    # rewrite backbone
    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = "build_resnet"
    _C.MODEL.BACKBONE.DEPTH = 18
    _C.MODEL.BACKBONE.STR_DEPTH = "400MF"
    _C.MODEL.BACKBONE.LAST_STRIDE = 2
    _C.MODEL.BACKBONE.PRETRAIN_PATH = ""
    _C.MODEL.BACKBONE.PRETRAIN = True
    _C.MODEL.BN_TYPE = "SyncBN"

    # optim and min_lr(for cosine schedule)
    _C.SOLVER.MIN_LR = 1e-8
    _C.SOLVER.OPTIM_NAME = "SGD"
    _C.SOLVER.COSINE_DECAY_ITER = 0.7

    # SWA options
    _C.SOLVER.SWA = CN()
    _C.SOLVER.SWA.ENABLED = False
    _C.SOLVER.SWA.ITER = 10
    _C.SOLVER.SWA.PERIOD = 2
    _C.SOLVER.SWA.LR_START = 2.5e-6
    _C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
    _C.SOLVER.SWA.LR_SCHED = False

    # Knowledge Distill
    _C.MODEL.CENTERNET.KD = CN()
    _C.MODEL.CENTERNET.KD.ENABLED = False
    _C.MODEL.CENTERNET.KD.TEACHER_CFG = ["None", ]
    _C.MODEL.CENTERNET.KD.TEACHER_WEIGTHS = ["None", ]
    _C.MODEL.CENTERNET.KD.KD_WEIGHT = [10.0, ]
    _C.MODEL.CENTERNET.KD.KD_CLS_WEIGHT = [1.0, ]
    _C.MODEL.CENTERNET.KD.KD_WH_WEIGHT = [1.0, ]
    _C.MODEL.CENTERNET.KD.KD_REG_WEIGHT = [0.1, ]
    _C.MODEL.CENTERNET.KD.KD_CLS_INDEX = [1, ]
    _C.MODEL.CENTERNET.KD.NORM_WH = [False, ]
    _C.MODEL.CENTERNET.KD.KD_WITHOUT_LABEL = False

    # input config
    _C.INPUT.FORMAT = "RGB"