#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as op
import os
import numpy as np
import torch
from torch.utils.data import  Dataset
from torch.utils.data import   DataLoader
import SimpleITK as sitk
from glob import glob
import torchio as tio
import h5py as h5
from options.BaseOptions import BaseOptions
opt = BaseOptions().gather_options()

class Quartet_dataset(Dataset):
    def __init__(self, data_dir):
        
        self.data_dir       = data_dir
        subfilelist         = os.listdir(data_dir) # all hyperfines and high field image in h5 fields
        subfilelist         = sorted(subfilelist)
        print(subfilelist)
        print('\n \n h5 Files, len list', len(subfilelist))
        
        self.subfilelist    = subfilelist 
        self.num_samples    = len(subfilelist )
        self.images =[]
        self.gts = []
        self.load(self.data_dir,self.subfilelist)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        LF_triplet = self.images[idx]
        HF_image   = self.gts[idx]
        
        subject = tio.Subject(
        # tio.ScalarImage stores a 4D tensor whose voxels encode signal intensity
        images_LF = tio.ScalarImage(tensor=LF_triplet),
        image_HF = tio.ScalarImage(tensor=HF_image) 
        )
        
        spatial_transforms = {
        tio.RandomAffine(scales=0.1, degrees=10,translation = 5): 0.5, 
        tio.RandomElasticDeformation(): 0.5,
        }
        
        # random bias field
        # random gamma
        transform = tio.Compose([
        tio.OneOf(spatial_transforms, p=0.5)
        #tio.RandomBiasField(),
        #tio.RandomGamma(log_gamma=(-0.3, 0.3)) 
        ])
        
        transformtmp= tio.RescaleIntensity(out_min_max=((0, 1)))
        transformed   =  transformtmp(subject)
        augmented     =  transform(transformed)
        tf_LF_triplet =  augmented.images_LF.numpy()
        tf_HF_image   =  augmented.image_HF.numpy()
 
        return tf_LF_triplet,tf_HF_image 
    
    def load(self, data_dir,subfilelist):
        LF_triplet  = []
        HF_image    = []
        print('Loading ' + str(self.num_samples) +' quartets \n \n')
        for i in range(self.num_samples):
           
            with h5.File(op.join(data_dir,subfilelist[i]), 'r') as f:
              imgAXI = f['image_axi'][()] # as np array
              imgCOR = f['image_cor'][()] # as np array
              imgSAG = f['image_sag'][()] # as np array
              imgHF  = f['image_gt'][()] # as np array
        
            X = np.stack([ imgAXI.astype(dtype='float32') ,  imgCOR.astype(dtype='float32'),  imgSAG.astype(dtype='float32') ], axis=0)
            Y = imgHF.astype(dtype='float32')

            LF_triplet.append(X)
            HF_image.append(Y)

        images = np.array(LF_triplet)
        gts = np.array(HF_image)
        gts = gts[:,None,:,:,:]
        print('\n rm test, size LF images',np.shape(images))  
        print('\n rm test, size HF image',np.shape(gts))  
        
        self.images = images
        self.gts = gts
        
    
if __name__ == '__main__':
     
     data_dir = opt.data_dir
     Dataset = Quartet_dataset(data_dir)  
     Y = torch.utils.data.DataLoader(Dataset, batch_size=4)       
    
        
    