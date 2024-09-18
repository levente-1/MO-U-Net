from options.BaseOptions import BaseOptions
import argparse


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--image_dir", type=str, default='/media/hdd/levibaljer/KhulaFinal/v2_Fold4/test/Sub63')
        parser.add_argument("--output_dir_pred", type=str, default='/media/hdd/levibaljer/KhulaFinal/v2_Fold4/test/Results_AS', help='path to the .nii result to save')
        parser.add_argument("--output_pref", type=str, default='63', help='prefix of the output file')

        return parser
    
    def gather_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        return opt