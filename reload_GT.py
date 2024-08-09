import torch
import os 
import SimpleITK as sitk
import torchio as tio
import numpy as np
import nibabel as nib

from options.TestOptions import TestOptions
opt = TestOptions().gather_options()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# Function below loads images from a single subject
def Image_loader(Image_dir):

    # Retrieve all files in subject folder
    subject_files = os.listdir(Image_dir)
    
    # Create list of full paths to subject images and sort them (this will order them as such: axial, coronal, sagittal)
    t2_file_list = [os.path.join(Image_dir,x) for x in subject_files]
    t2_file_list.sort()
    for i in t2_file_list[:]:
        if 'GT' not in i:
            t2_file_list.remove(i)
    
    print(t2_file_list)

    # Create list to hold the 3 input images for selected subject (as a PyTorch tensor)
    image_tensors = []

    # Loop through each file path
    for file in t2_file_list:
        
        # Load using TorchIO (package that allows for all the preprocessing steps below)
        subject = tio.Subject(t2=tio.ScalarImage(file))
        
        # Apply padding as necessary 
        # This will ensure that the shape of the input file is a perfect cube; allows the UNet to process more easily
        edge_max = max(subject.t2.data.shape)
        padding = ((edge_max - subject.t2.data.shape[1]) // 2, 
                    (edge_max - subject.t2.data.shape[2]) // 2,
                        (edge_max - subject.t2.data.shape[3]) // 2)
        
        # In addition to padding, we resample the image to be 1mm isotropic 
        transform_1 = tio.Compose([
            tio.Pad(padding),
            tio.transforms.Resample((1.6,1.6,1.6))
        ])
        
        # Apply transforms outlined above
        subject = transform_1(subject)

        # Finally, rescale voxel intensity to be between 0 and 1
        transform_2 = tio.RescaleIntensity(out_min_max=((0, 1)))
        subject    = transform_2(subject)
        
        # Make sure data is of right shape to save as a tensor
        image_tensor = subject.t2.data.unsqueeze(0).float()
        image_tensors.append(image_tensor)
    
    # Return list that contains all image tensors (axial, coronal, sagittal)
    return image_tensors, t2_file_list

# Run function above with selected directory
Image_tensors, file_list = Image_loader(opt.image_dir)

# Concatenate tensors (instead of three separate inputs of shape (1, 160, 160, 160) we have one single input (3, 160, 160, 160))
tensor_GT = Image_tensors[0]

# Select output directory to save images
output_dir = opt.output_dir_pred
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Saving prediction (the orientation of the images will be altered because of TorchIO)
high_field = tensor_GT.cpu().detach().numpy().astype(np.float32)
sitk.WriteImage(sitk.GetImageFromArray(high_field[0, 0, :, :, :]),
                        os.path.join(output_dir, '{}gt.nii.gz'.format(opt.output_pref)))


# Reload image (this will automatically reset orientation)
def reload(imdir):
    subject = tio.Subject(t2=tio.ScalarImage(imdir))
    image_tensor = subject.t2.data.unsqueeze(0).float()
    return image_tensor

tio_dir = os.path.join(output_dir, '{}gt.nii.gz'.format(opt.output_pref))
tio_tensor = reload(tio_dir)

# Resample image to original resolution (1mm isotropic)
myTransform2 = tio.transforms.Resample((0.625,0.625,0.625))
tio_tensor = tio_tensor.squeeze(0)
tio_tensor = myTransform2(tio_tensor)

# Save resampled image
resampled = tio_tensor.cpu().detach().numpy().astype(np.float32)
sitk.WriteImage(sitk.GetImageFromArray(resampled[0, :, :, :]),
                            os.path.join(output_dir, '{}gt_resampled.nii.gz'.format(opt.output_pref)))

# Reload original axial image with nibabel
OriginalGT = nib.load(file_list[0])
nib.aff2axcodes(OriginalGT.affine)

# Set sform and qform of resampled image to be the same as the original axial image
finalPrediction = nib.load((os.path.join(output_dir, '{}gt_resampled.nii.gz'.format(opt.output_pref))))
finalPrediction.set_qform(OriginalGT.affine)
finalPrediction.set_sform(OriginalGT.affine)
nib.save(finalPrediction, os.path.join(output_dir, 'sub{}gt_final.nii'.format(opt.output_pref))) 
