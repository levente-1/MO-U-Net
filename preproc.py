#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from glob import glob

from matplotlib import image
from sklearn.utils import resample

from torch.utils.data import Dataset

import numpy as np
import os
import os.path as op
import glob
import sys
import logging
from tqdm import trange, tqdm
from tqdm.contrib.concurrent import thread_map
from multiprocessing import cpu_count
import torchio as tio
import h5py as h5
import matplotlib.pyplot as plt


# Retrieve subject folders - NOTE MUST NOT HAVE h5 folder there already or will confuse folder list
data_dir =  '/media/hdd/levibaljer/KhulaFinal'
list_subs = os.listdir(data_dir)
for i in list_subs:
    if 'Sub' not in i:
        list_subs.remove(i)
list_basenames = [op.join(data_dir,b) for b in list_subs]

list_basenames.sort()
# Retrieve folders for axial, coronal, sagittal, groundtruth images
list_images_axi = [op.join(x, "AXIextracted.nii") for x in list_basenames]
list_images_cor = [op.join(x, "CORextracted.nii") for x in list_basenames]
list_images_sag = [op.join(x, "SAGextracted.nii") for x in list_basenames]

list_images_gt = [op.join(x, "GTextracted.nii") for x in list_basenames]

#list_basenames = _list_basenames
print(list_basenames)
num_samples = len(list_basenames)
 
def _load(x):
    '''
    Load each MRI
    '''
    img_axi, img_cor, img_sag, img_gt, basename = x
    
    # tio.Subject stores all subject's data in a dictionary
    subject = tio.Subject(
        # tio.ScalarImage stores a 4D tensor whose voxels encode signal intensity
        image_axi = tio.ScalarImage(img_axi),
        image_cor = tio.ScalarImage(img_cor),
        image_sag = tio.ScalarImage(img_sag),
        image_gt = tio.ScalarImage(img_gt)
    )
    
    # tio.Resample to resample into 1mm isotropic
    transform_1 = tio.Compose([
        tio.transforms.Resample((1.,1.,1.))
    ])
    
    # Apply transform to all subjects (and all images within subject)
    #subject = transform_1(subject)
    # Padding applied using sample axial image 
    edge_max = max(subject.image_axi.data.shape)
    padding = ((edge_max - subject.image_axi.data.shape[1]) // 2, 
                (edge_max - subject.image_axi.data.shape[2]) // 2,
                    (edge_max - subject.image_axi.data.shape[3]) // 2)

    transform_2 = tio.Compose([
        tio.Pad(padding),
        tio.transforms.Resample((1.6,1.6,1.6))
    ])
    # Apply transform2 to all subjects (and all images within subject)
    subject = transform_2(subject)
    
    transform3 = tio.RescaleIntensity(out_min_max=((0, 1)))
    subject    = transform3(subject)
    
    preprocessed_path = op.join(data_dir, "preprocessed_h5_Khula")
    print(basename)
    if not op.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    with h5.File(op.join(preprocessed_path, basename + '.h5'), 'w') as f:
        f.create_dataset('image_axi', data=subject.image_axi.data[0])
        f.create_dataset('image_cor', data=subject.image_cor.data[0])
        f.create_dataset('image_sag', data=subject.image_sag.data[0])
        f.create_dataset('image_gt', data=subject.image_gt.data[0])
      
def load(list_images_axi,list_images_cor, list_images_sag, list_images_gt, list_basenames):
    thread_map(_load, zip(list_images_axi, list_images_cor, list_images_sag, list_images_gt,list_basenames), max_workers=1, total=num_samples)

load(list_images_axi, list_images_cor, list_images_sag, list_images_gt,list_basenames)

