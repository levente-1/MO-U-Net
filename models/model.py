import torch.nn as nn
import torch
import torch.optim as optim
from torchmetrics import PeakSignalNoiseRatio #StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
import lpips
import numpy as np

from models.buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, \
    create_decoders
#from .utils import number_of_features_per_level, get_class
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(
                f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
           
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x

    
 


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)
        
        self.perc_loss = lpips.LPIPS(net='alex').to(device)

        for ll in range(5):
            tensor = self.perc_loss.lins[ll].model[1].weight.flatten()
            for i in tensor:
                if i < 0:
                    print("NEGATIVE WEIGHTS IN NETWORK") 
        print("NO NEGATIVE WEIGHTS IN NETWORK")

    def loss(self,im,im_hat):

        def randomSlices(img_gt, img_pred):
            dim1_gt = []
            dim2_gt = []
            dim3_gt = []

            dim1_pred = []
            dim2_pred = []
            dim3_pred = []

            for i in range(2):
                dim1Slice = np.random.randint(50, 120)
                dim2Slice = np.random.randint(40, 110)
                dim3Slice = np.random.randint(40, 100)

                dim1_gt.append(img_gt[:, :, dim1Slice, :, :])
                dim2_gt.append(img_gt[:, :, :, dim2Slice, :])
                dim3_gt.append(img_gt[:, :, :, :, dim3Slice])

                dim1_pred.append(img_pred[:, :, dim1Slice, :, :])
                dim2_pred.append(img_pred[:, :, :, dim2Slice, :])
                dim3_pred.append(img_pred[:, :, :, :, dim3Slice])

            finalList_gt = dim1_gt + dim2_gt + dim3_gt
            finalList_pred = dim1_pred + dim2_pred + dim3_pred
        
            torchList_gt = []
            torchList_pred = []

            for i in range(len(finalList_gt)):
                new_torch_gt = (finalList_gt[i] * 2) - 1
                new_torch_pred = (finalList_pred[i] * 2) - 1
                
                new_torch_gt = new_torch_gt.expand(1, 3, 160, 160)
                new_torch_pred = new_torch_pred.expand(1, 3, 160, 160)
            
                torchList_gt.append(new_torch_gt)
                torchList_pred.append(new_torch_pred)
        
            finalTorchGT = torch.concat(torchList_gt, dim=0)
            finalTorchpred = torch.concat(torchList_pred, dim=0)
            
            return finalTorchGT, finalTorchpred
          
        mse = torch.nn.MSELoss().to(device)
        loss_mse = mse(im,im_hat)
        # psnr = PeakSignalNoiseRatio(data_range = 1.0).to(device)
        # loss_now = 1 - psnr(im_hat,im)
        # l1 = torch.nn.L1Loss()

        im = im.detach()
        im_hat = im_hat.detach()
        slicesGT, slicesPred = randomSlices(im, im_hat)
        with torch.no_grad():
            slicesGT = slicesGT.to(device)
            slicesPred = slicesPred.to(device)

        perc = self.perc_loss(slicesGT, slicesPred)
        perc_mean = torch.mean(perc)

        loss_now = 100*loss_mse + perc_mean
        # print("LPIPS loss: " + str(perc_mean), "MSE loss: " + str(loss_mse), "Total loss: " + str(loss_now))

        return loss_now
            

 

def number_of_features_per_level(init_channel_number, num_levels):
  return [init_channel_number * 2 ** k for k in range(num_levels)]










class ResidualUNet3D(Abstract3DUNet):
   """
   Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
   Uses ExtResNetBlock as a basic building block, summation joining instead
   of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
   Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
   """

   def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=64, layer_order='gcr',
                num_groups=8, num_levels=5, is_segmentation=False, conv_padding=1, **kwargs):
       super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            final_sigmoid=final_sigmoid,
                                            basic_module=ExtResNetBlock,
                                            f_maps=f_maps,
                                            layer_order=layer_order,
                                            num_groups=num_groups,
                                            num_levels=num_levels,
                                            is_segmentation=is_segmentation,
                                            conv_padding=conv_padding,
                                            **kwargs)


class UNet2D(Abstract3DUNet):
   """
   Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
   """

   def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
       if conv_padding == 1:
           conv_padding = (0, 1, 1)
       super(UNet2D, self).__init__(in_channels=in_channels,
                                    out_channels=out_channels,
                                    final_sigmoid=final_sigmoid,
                                    basic_module=DoubleConv,
                                    f_maps=f_maps,
                                    layer_order=layer_order,
                                    num_groups=num_groups,
                                    num_levels=num_levels,
                                    is_segmentation=is_segmentation,
                                    conv_kernel_size=(1, 3, 3),
                                    pool_kernel_size=(1, 2, 2),
                                    conv_padding=conv_padding,
                                    **kwargs)


# def get_model(model_config):
#     model_class = get_class(model_config['name'], modules=[
#                             'pytorch3dunet.unet3d.model'])
#     return model_class(**model_config)
