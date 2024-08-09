import argparse
import os
import pandas as pd

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--data_dir', default = 'Your path here', help='path to data directory')
        parser.add_argument('--preproc_train', default = '/media/hdd/levibaljer/KhulaFinal/Fold2/train/preprocessed_h5_train', help='path to preprocessed h5 file (training set)')
        parser.add_argument('--preproc_val', default = '/media/hdd/levibaljer/KhulaFinal/Fold2/val/preprocessed_h5_val', help='path to preprocessed h5 file (validation set)')
        parser.add_argument('--output_dir', default = '/media/hdd/levibaljer/UNet_old/Fold2_LPIPS', help='path to store model checkpoints and predictions')
        parser.add_argument('--batch_size', default = 1, help='batch size')
        parser.add_argument('--num_epochs', default = 1000, help='number of training epochs')
        parser.add_argument('--n_splits', default = 10, help='number of splits for kfold cross validation')
        parser.add_argument('--checkpoint', default=None, help='path to model checkpoint (None if training from scratch, otherwise /media/hdd/levibaljer/UNet_old/Fold1_L2/checkpoints)')
        parser.add_argument('--id_path', default = 'Your path here', help='path to save train and test ids')
        self.initialized = True
        return parser
    
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()

        return opt