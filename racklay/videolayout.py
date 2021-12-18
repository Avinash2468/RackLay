from collections import OrderedDict

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import ResnetEncoder

# Utils


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Encoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_layers : int
        Number of layers to use in the ResNet
    img_ht : int
        Height of the input RGB image
    img_wt : int
        Width of the input RGB image
    pretrained : bool
        Whether to initialize ResNet with pretrained ImageNet parameters

    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """

    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Encoder, self).__init__()

        # opt.weights_init == "pretrained"))
        self.resnet_encoder = ResnetEncoder(num_layers, pretrained)
        num_ch_enc = self.resnet_encoder.num_ch_enc
        # convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128) 
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of Image tensors
            | Shape: (batch_size, 3, img_height, img_width)

        Returns
        -------
        x : torch.FloatTensor
            Batch of low-dimensional image representations
            | Shape: (batch_size, 128, img_height/128, img_width/128)
        """

        batch_size, c, h, w = x.shape

        x = self.resnet_encoder(x)[-1]

        #x = self.pool(self.conv1(x))
        #x = self.conv2(x)
        #x = self.pool(x)  

        return x


class Decoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_ch_enc : list
        channels used by the ResNet Encoder at different layers

    Methods
    -------
    forward(x, ):
        Processes input image features into output occupancy maps/layouts
    """

    def __init__(self, num_ch_enc, num_out_ch, oct_map_size):
        super(Decoder, self).__init__()
        self.num_output_channels = num_out_ch
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256, 512]) 
        self.oct_map_size = oct_map_size
        self.pool = nn.MaxPool2d(2)
        # decoder
        self.convs = OrderedDict()
        for i in range(5, -1, -1):
            # upconv_0
            num_ch_in = 512 if i == 5 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(
            self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, x, is_training=True):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of encoded feature tensors
            | Shape: (batch_size, 128, occ_map_size/2^5, occ_map_size/2^5)
        is_training : bool
            whether its training or testing phase

        Returns
        -------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)
        """

        for i in range(5, -1, -1):

            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)

        x = self.pool(x)

        if is_training:
            x = self.convs["topview"](x)
        else:
            softmax = nn.Softmax2d()
            x = softmax(self.convs["topview"](x))

        return x


class Discriminator(nn.Module):
    """
    A patch discriminator used to regularize the decoder
    in order to produce layouts close to the true data distribution
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, 8, 3, 2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(8, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of output Layouts
            | Shape: (batch_size, 2, occ_map_size, occ_map_size)

        Returns
        -------
        x : torch.FloatTensor
            Patch output of the Discriminator
            | Shape: (batch_size, 1, occ_map_size/16, occ_map_size/16)
        """

        return self.main(x)

class VideoLayout(nn.Module):
    """docstring for MonoLayout"nn.Module def __init__(self, arg):
        super(MonoLayout,nn.Module.__init__()
        self.arg = arg
    """

    def __init__(self, opt, num_ch_dec=2):
        super(VideoLayout, self).__init__()
        self.opt = opt
        self.criterion_d = nn.BCEWithLogitsLoss()
        # self.encoder = Encoder(18, opt.height, opt.width, pretrained=True)
        
        self.encoder = Encoder(18, self.opt.height, self.opt.width, True)

        self.convlstm = ConvLSTM((8, 8), 128, 128, (3, 3), 1)

        if self.opt.type == "both":
            self.top_decoder = Decoder(
                self.encoder.resnet_encoder.num_ch_enc, 3*self.opt.num_racks, self.opt.occ_map_size)
            self.top_discr = Discriminator()
            self.front_discr = Discriminator()
            self.front_decoder = Decoder(
                self.encoder.resnet_encoder.num_ch_enc, 3*self.opt.num_racks,self.opt.occ_map_size)
            self.parameters = list(self.encoder.parameters()) + list(self.convlstm.parameters())\
                             + list(self.top_decoder.parameters()) + list(self.front_decoder.parameters())
            self.parameters_to_train_D = list(self.top_discr.parameters()) + list(self.front_discr.parameters())

        elif self.opt.type == "topview":
            self.top_decoder = Decoder(
                self.encoder.resnet_encoder.num_ch_enc, 3*self.opt.num_racks,self.opt.occ_map_size)
            self.top_discr = Discriminator()
            self.parameters = list(self.encoder.parameters()) + list(self.convlstm.parameters())\
                             + list(self.top_decoder.parameters())
            self.parameters_to_train_D = list(self.top_discr.parameters())

        elif self.opt.type == "frontview":
            self.front_decoder = Decoder(
                self.encoder.resnet_encoder.num_ch_enc, 3*self.opt.num_racks,self.opt.occ_map_size)
            self.front_discr = Discriminator()
            self.parameters = list(self.encoder.parameters()) + list(self.convlstm.parameters())\
                             + list(self.front_decoder.parameters())
            self.parameters_to_train_D = list(self.front_discr.parameters())

        self.model_optimizer = optim.Adam(self.parameters, self.opt.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.discr_optimizer = optim.Adam(self.parameters_to_train_D, self.opt.lr)
        self.discr_lr_scheduler = optim.lr_scheduler.StepLR(
                self.discr_optimizer, self.opt.scheduler_step_size, 0.1)

        self.patch = (1, self.opt.occ_map_size // 2 **
                      4, self.opt.occ_map_size // 2**4)

        self.bce = nn.BCEWithLogitsLoss()

        self.valid = Variable(
            torch.Tensor(
                np.ones(
                    (self.opt.batch_size,
                     *self.patch))),
            requires_grad=False).float().cuda()
        self.fake = Variable(
            torch.Tensor(
                np.zeros(
                    (self.opt.batch_size,
                     *self.patch))),
            requires_grad=False).float().cuda()

    def forward(self, x, is_training=True):
        outputs = {}
        mu = self.encoder(x)
        # z = self.reparameterize(is_training, mu, logvar)
        z = self.convlstm(mu)[0][0][:,-1]
        if self.opt.type == "both":
            outputs["topview"] = self.top_decoder(z)
            outputs["frontview"] = self.front_decoder(z)
        elif self.opt.type == "topview":
            outputs["topview"] = self.top_decoder(z)
        elif self.opt.type == "frontview":
            outputs["frontview"] = self.front_decoder(z) 
        return outputs

    def reparameterize(self, is_training, mu, logvar):
        if is_training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def step(self, inputs, outputs, losses, epoch):
        self.model_optimizer.zero_grad()
        self.discr_optimizer.zero_grad()
        loss = {}
        if(self.opt.type == "both" or self.opt.type == "topview"):
            loss_D_top = 0
            loss_G_top = 0
            for i in range(self.opt.num_racks): # For top view
                gen_temp = outputs["topview"][:,3*i:3*i+3,:,:]
                gen_temp = torch.argmax(gen_temp, 1)
                gen_temp = torch.unsqueeze(gen_temp, 1).float()
                true_temp = inputs["topview"].float()[:,i,:,:]
                true_temp = torch.unsqueeze(true_temp, 1).float()
                fake_pred = self.top_discr(gen_temp)
                real_pred = self.top_discr(true_temp)
                loss_GAN = self.criterion_d(fake_pred, self.valid)
                loss_D_top += self.criterion_d(
                    fake_pred, self.fake) + self.criterion_d(real_pred, self.valid)
                loss_G_top += self.opt.lambda_D * loss_GAN + losses["top_loss"]
            loss_G_top.backward(retain_graph=True)                
            loss_D_top.backward(retain_graph=True)
            loss["loss_G_top"] = loss_G_top
            loss["loss_D_top"] = loss_D_top

        if(self.opt.type == "both" or self.opt.type == "frontview"):
            loss_D_front = 0
            loss_G_front = 0
            for i in range(self.opt.num_racks): # For front view
                gen_temp = outputs["frontview"][:,3*i:3*i+3,:,:]
                gen_temp = torch.argmax(gen_temp, 1)
                gen_temp = torch.unsqueeze(gen_temp, 1).float()
                true_temp = inputs["frontview"].float()[:,i,:,:]
                true_temp = torch.unsqueeze(true_temp, 1).float()
                fake_pred = self.front_discr(gen_temp)
                real_pred = self.front_discr(true_temp)
                loss_GAN = self.criterion_d(fake_pred, self.valid)
                loss_D_front += self.criterion_d(
                    fake_pred, self.fake) + self.criterion_d(real_pred, self.valid)
                loss_G_front += self.opt.lambda_D * loss_GAN + losses["front_loss"]
            loss_G_front.backward(retain_graph=True)
            loss_D_front.backward()
            loss["loss_G_front"] = loss_G_front
            loss["loss_D_front"] = loss_D_front

        self.model_optimizer.step()
        self.discr_optimizer.step()
        
        return loss