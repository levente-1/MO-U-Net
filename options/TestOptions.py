from BaseOptions import BaseOptions
import argparse


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--image", type=str, default='./Data_folder/test/images/0.nii')
        parser.add_argument("--output_dir", type=str, default='./Data_folder/test/images/result_0.nii', help='path to the .nii result to save')
        parser.add_argument("--output_pref", type=str, default='./Data_folder/test/images/0.nii', help='prefix of the output file')

        return parser
    
    def gather_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        return opt