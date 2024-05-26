from easydict import EasyDict
from torch import nn
from model import *
from dataset import GraphDataset
from HWrunner import HomeworkRunner

CFG = EasyDict()

CFG.DESCRIPTION = "HW7 OverSmooth configuration"
CFG.GPU_NUM = 1

CFG.RUNNER = HomeworkRunner

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "simpleGCN2"
CFG.MODEL.ARCH = simpleGCN2

CFG.MODEL.PARAM = {
    "numlayer":20,
    "inchannel":745,
    "midchannel":512,
    "outchannel":8,
    "isNormalize":True,
    "isshared_weights":True,
    "isResidual":False
}
CFG.TRAIN = EasyDict()
CFG.TRAIN.DATASET = EasyDict()
CFG.TRAIN.DATASET.ARCH = GraphDataset

dataset = {
    "root":"data",
    "name":"photo",
    "type":"Amazon",
    "isTrain":True,
    "maskRate":0.2
}
CFG.TRAIN.DATASET.PARAM = dataset

CFG.TRAIN.LOSS = nn.CrossEntropyLoss()

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}

EPOCHNUM=200
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "T_max":EPOCHNUM,
    "eta_min":1e-5,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

CFG.TRAIN.NUM_EPOCHS = EPOCHNUM
CFG.TRAIN.CKPT_SAVE_DIR = "experiment"
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.DATA.BATCH_SIZE = 1
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= Test ================= #
CFG.VAL = EasyDict()
CFG.VAL.DATASET=EasyDict()
CFG.VAL.DATASET.ARCH = GraphDataset
dataset['isTrain']=False
CFG.VAL.DATASET.PARAM = dataset

CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 1
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True
