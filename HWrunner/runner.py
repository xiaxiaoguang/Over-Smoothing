import torch
from easytorch import Runner
from typing import Dict, Tuple, Union
import torch.nn as nn
from torch.utils.data import Dataset

from metric import Eng,accuracy

class HomeworkRunner(Runner):
    def __init__(self,cfg:dict):
        super().__init__(cfg)
        self.register_epoch_meter("train_Loss", "train", "{:.4f}")
        self.register_epoch_meter("val_Loss", "val", "{:.4f}")
        self.register_epoch_meter("val_Accuracy", "val", "{:.4f}")
        self.register_epoch_meter("val_Dirichlet", "val", "{:.5f}")
        self.loss = cfg["TRAIN"]["LOSS"]
        self.logger.info(self.model)

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        return cfg["MODEL"]["ARCH"](**cfg.MODEL.PARAM)
    
    @staticmethod
    def build_train_dataset(cfg: Dict) -> Dataset:
        return cfg["TRAIN"]["DATASET"]["ARCH"](**cfg.TRAIN.DATASET.PARAM)
    
    @staticmethod
    def build_val_dataset(cfg:Dict)->Dataset:
        return cfg["VAL"]["DATASET"]["ARCH"](**cfg.VAL.DATASET.PARAM)
    
    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        edg = self.to_running_device(data[2].squeeze(0))
        input = self.to_running_device(data[0])
        y = self.to_running_device(data[1].squeeze(0))
        index = self.to_running_device(data[3].squeeze(0))
        _,output = self.model(input,edg)

        x = output.squeeze(0)[index,:]
        y = y[index]

        loss = self.loss(x,y)

        self.update_epoch_meter("train_Loss", loss.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):        
        edg = self.to_running_device(data[2].squeeze(0))
        input = self.to_running_device(data[0])
        y = self.to_running_device(data[1].squeeze(0))
        index = self.to_running_device(data[3].squeeze(0))        
        hid,otp = self.model(input,edg)

        hid = hid.squeeze(0)
        pre = otp.squeeze(0)[index,:]
        y   = y[index]

        energy = Eng(hid,edg)
        loss = self.loss(pre,y)
        acu = accuracy(pre,y)

        self.update_epoch_meter("val_Loss",loss.item())
        self.update_epoch_meter("val_Dirichlet",energy.item())
        self.update_epoch_meter("val_Accuracy",acu.item())
        return loss
