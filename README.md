# Official implementation of Multi-orientation U-Net

Pytorch pipeline for 3D image domain translation using Multi-orientation U-Net. Code-base adapted from:

- https://github.com/ExtremeViscent/SR-UNet

### Prerequisites

See `requirements.txt`

```
pip install -r requirements.txt
```

### Preprocessing

Before training the model, set data_dir in "Base_options.py" and run "run preproc.py" to convert data into h5 file format (required format for the dataloader). Files should be arranged in the following format prior to running the preprocessing script:

	├── Data_folder                   
	|   ├── Subject_1               
	|   |   ├── Axial.nii 
    |   |   ├── Coronal.nii 
    |   |   ├── Sagittal.nii
	|   |   └── High_field.nii                   
	|   ├── Subject_2                       
	|   |   ├── Axial.nii 
    |   |   ├── Coronal.nii 
    |   |   ├── Sagittal.nii
	|   |   └── High_field.nii  

### Training

Modify "BaseOptions.py" to set directory for preprocessed data and training configurations. Training and test sets are split on the fly, with their respective IDs saved as train_ids.npy & test_ids.npy in the pre-specified "id_path" directory.

### Prediction

Modify "TestOptions.py" file to specify input image, output directory, and prefix specifying subject number. 
Run "test.py" to obtain prediction (saved as 'sub<x>pred_final.nii')




