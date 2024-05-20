import argparse
import os
import pandas as pd

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def intialize(self, parser):
        parser.add_argument('--data_dir', default = 'Your path here', help='path to data directory')
        parser.add_argument('--preproc_dir', default = 'Your path here', help='path to preprocessed h5 file')
        parser.add_argument('--output_dir', default = 'Your path here', help='path to store model checkpoints and predictions')
        parser.add_argument('--batch_size', default = 1, help='batch size')
        parser.add_argument('--num_epochs', default = 1000, help='number of training epochs')
        parser.add_argument('--n_splits', default = 10, help='number of splits for kfold cross validation')
        parser.add_argument('--checkpoint', default=None, help='path to model checkpoint (None if training from scratch)')
        parser.add_argument('--id_path', default = 'Your path here', help='path to save train and test ids')
        self.initialized = True
        return parser
    
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.intialize(parser)
        opt, _ = parser.parse_known_args()

        return opt