from cmath import phase
from datetime import date
from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F


from dataloaders import get_quartet_dataloader
from models.model import UNet3D
import importlib
import SimpleITK as sitk
from PIL import Image
from options.BaseOptions import BaseOptions

from torch.utils.tensorboard import SummaryWriter
import os
import os.path as op
# import time
from tqdm import tqdm
import numpy as np

opt = BaseOptions().gather_options()

train_dir = opt.preproc_train
val_dir = opt.preproc_val
OUTPUT_DIR = opt.output_dir
batch_size = opt.batch_size
n_splits = opt.n_splits
checkpoint = opt.checkpoint
num_epochs = opt.num_epochs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TensorBoardLogger():
    def __init__(self, log_dir, **kwargs):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir, **kwargs)
        
    
    def __call__(self, phase, step, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_scalar(f'{key}/{phase}', value, step)

class BetaScheduler():
    def __init__(self, model, min=0,max=0.0001, cycle_len=1000):
        self.model = model
        self.min = min
        self.max = max
        self.current_step = 0
        self.cycle_len = cycle_len
    def get_beta(self):
        return self.model.alpha
    def step(self):
        self.model.alpha = self.min + (self.max - self.min) * (1 - np.cos(self.current_step / self.cycle_len * np.pi)) / 2
        self.current_step += 1 



def eval(model, cur_epoch):
    output_dir=os.path.join(OUTPUT_DIR)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    pred_dir = os.path.join(output_dir, 'preds')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
   
    ckpt_path = os.path.join(checkpoint_dir, '{}.pth'.format(cur_epoch))
    if cur_epoch % 10 == 0:
        torch.save(model,ckpt_path)
  
    with torch.no_grad():
        image = np.load(os.path.join(output_dir, 'test_image.npy'))
        target = np.load(os.path.join(output_dir, 'test_target.npy'))
        image = torch.tensor(image).unsqueeze(0).cuda()
        target = torch.tensor(target).unsqueeze(0).cuda()
        model.eval()
        pred = model(image)
        image = image.cpu().detach().numpy().astype(np.float32)
        target = target.cpu().detach().numpy().astype(np.float32)
        pred = pred.cpu().detach().numpy().astype(np.float32)
        im_pred = pred[0, 0, pred.shape[2]//2, :, :]
        im_pred = (im_pred-np.min(im_pred)) / \
            (np.max(im_pred)-np.min(im_pred))*255
        im_pred = Image.fromarray(im_pred).convert('RGB')
        if cur_epoch == 0:
            
            sitk.WriteImage(sitk.GetImageFromArray(image[0, 0, :, :, :]),
                            os.path.join(output_dir, 'image.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(target[0, :, :, :]),
                            os.path.join(output_dir, 'target.nii.gz'))
            im_image = image[0, 0, image.shape[2]//2, :, :]
            im_target = target[0, 0, image.shape[2]//2, :, :]
            im_image = (im_image-np.min(im_image)) / \
                (np.max(im_image)-np.min(im_image))*255
            im_target = (im_target-np.min(im_target)) / \
                (np.max(im_target)-np.min(im_target))*255
            im_image = Image.fromarray(im_image).convert('RGB')
            im_target = Image.fromarray(im_target).convert('RGB')
            im_image.save(os.path.join(output_dir, 'image.png'))
            im_target.save(os.path.join(output_dir, 'target.png'))

        if not os.path.exists(os.path.join(pred_dir, '2d')):
            os.makedirs(os.path.join(pred_dir, '2d'))
        if cur_epoch % 10 == 0:
            im_pred.save(os.path.join(pred_dir, "2d",
                        '{}.png'.format(cur_epoch)))

def train():
    torch.autograd.set_detect_anomaly(True)

    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader, val_loader = get_quartet_dataloader(train_dir = train_dir, val_dir = val_dir, batch_size = batch_size, output_dir=output_dir)
    
    if checkpoint is not None:
        model = torch.load((checkpoint))
    else:
        model =  UNet3D(in_channels=3, out_channels=1) # default args

    optim = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999)
    )
    TBLogger = TensorBoardLogger(log_dir = op.join(output_dir, 'tb_logs'))
    sample_val_image,sample_val_target = val_loader.dataset.__getitem__(0)
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))
    np.save(os.path.join(output_dir, 'test_image.npy'), sample_val_image)
    np.save(os.path.join(output_dir, 'test_target.npy'), sample_val_target)

    model.cuda()
    n_step = 0
    n_step_val = 0

    losses = []
    val_losses = []
    for epoch in range(num_epochs):
        losses.append(0)
        model.train()
        
        for im,gt in tqdm(train_loader):
            im=im.cuda()
            gt=gt.cuda()
            optim.zero_grad()
            output = model(im)
            loss = model.loss(gt, output)

            try:
                loss.backward()
                optim.step()
            except Exception as e:
                pass

            n_step+=1
            losses[-1] += loss.item()
        TBLogger(phase='train', step=epoch, loss=losses[-1])

        model.eval()
        val_losses.append(0)

        with torch.no_grad():
            for im,gt in tqdm(val_loader):
                im=im.cuda()
                gt=gt.cuda()
                output = model(im)  
                loss = model.loss(gt, output)
                n_step_val+=1
                val_losses[-1] += loss.item()
        TBLogger(phase='val', step=epoch,loss=val_losses[-1])

        eval(model,epoch)

            

if __name__ == '__main__':
    train()
