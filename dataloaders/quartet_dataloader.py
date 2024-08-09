from sklearn import datasets
import torch
import torch.distributed as dist
#from datasets import SynthDolphinDataset 
from quartet_dataset import Quartet_dataset
from sklearn.model_selection import KFold
import numpy as np
import os
import torchio as tio
from options.BaseOptions import BaseOptions
opt = BaseOptions().gather_options()

def get_dataloader(train_dir=None, val_dir=None, batch_size:int=1,output_dir="./output",**kwargs):
    dataset_train = Quartet_dataset(train_dir)
    dataset_val = Quartet_dataset(val_dir)
  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trainloader = torch.utils.data.DataLoader(
                    dataset_train, 
                    batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(
                    dataset_val,
                    batch_size=batch_size)
        
    return trainloader, valloader
