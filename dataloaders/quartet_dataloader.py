from sklearn import datasets
import torch
import torch.distributed as dist
#from datasets import SynthDolphinDataset 
from quartet_dataset import Quartet_dataset
from sklearn.model_selection import KFold
import numpy as np
import os
import torchio as tio

def get_dataloader(data_dir=None,batch_size:int=1,output_dir="./output", n_splits = 5,**kwargs):
    dataset = Quartet_dataset()
    kfold = KFold(n_splits=n_splits, shuffle=True)
    loaders = []
  
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
      
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'train_ids_{}.npy'.format(fold)), train_ids)
        np.save(os.path.join(output_dir, 'test_ids_{}.npy'.format(fold)), test_ids)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size,
                        sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, 
                        sampler=test_subsampler)
        loaders.append((trainloader, testloader))
        

        np.save(os.path.join(output_dir, 'train_ids_{}.npy'.format(fold)), train_ids)
        np.save(os.path.join(output_dir, 'test_ids_{}.npy'.format(fold)), test_ids)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size,
                        sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, 
                        sampler=test_subsampler)
        loaders.append((trainloader, testloader))
        
    return loaders
