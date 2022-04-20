import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from pytorch_lightning import LightningModule
from omegaconf import DictConfig

import numpy as np

class CNN_Encoder(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.input_size = (cfg.arch.cnn_enc.in_channels, cfg.arch.cnn_enc.input_height, cfg.arch.cnn_enc.input_width)
        self.channel_mult = cfg.arch.cnn_enc.channel_mult 

        #convolutions
        self.conv = nn.Sequential(
            #1
            nn.Conv2d(in_channels=cfg.arch.cnn_enc.in_channels,
                      out_channels=self.channel_mult*1,
                      kernel_size=cfg.arch.cnn_enc.kernel_size,
                      stride=cfg.arch.cnn_enc.stride,
                      padding=cfg.arch.cnn_enc.padding),
            nn.LeakyReLU(0.2, inplace=True),
            #2
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2,
                      cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            #3
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4,
                      cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            #4
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8,
                      cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            #5
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, cfg.arch.cnn_enc.padding),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, cfg.arch.cnn_enc.output_size),
            nn.BatchNorm1d(cfg.arch.cnn_enc.output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size).float())
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.output_height = cfg.arch.cnn_dec.output_height
        self.output_width = cfg.arch.cnn_dec.output_width
        self.input_dim = cfg.arch.embedding_size
        self.channel_mult = cfg.arch.cnn_dec.channel_mult
        self.output_channels = cfg.arch.cnn_dec.out_channels
        self.fc_output_dim = cfg.arch.cnn_dec.fc_output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            #1
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4,
                               (cfg.arch.cnn_dec.kernel_size,
                                cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor),
                               cfg.arch.cnn_dec.stride,
                               cfg.arch.cnn_dec.padding,
                               bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            # state size. self.channel_mult*32 x 4 x 4
            #2
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                               (cfg.arch.cnn_dec.kernel_size + 1,
                                cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                               cfg.arch.cnn_dec.stride,
                               cfg.arch.cnn_dec.padding, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            # state size. self.channel_mult*16 x 7 x 7
            #3
            nn.ConvTranspose2d(self.channel_mult*2, self.output_channels, 
                               (cfg.arch.cnn_dec.kernel_size + 1,
                                cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                               cfg.arch.cnn_dec.stride,
                               cfg.arch.cnn_dec.padding, bias=False),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(True),
            # state size. self.channel_mult*8 x 14 x 14
            #4
            nn.ConvTranspose2d(self.output_channels, self.output_channels, 
                               (cfg.arch.cnn_dec.kernel_size + 1,
                                cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                               cfg.arch.cnn_dec.stride,
                               cfg.arch.cnn_dec.padding, bias=False),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(True),
            # state size. self.channel_mult*8 x 14 x 14
            #5
            nn.ConvTranspose2d(self.output_channels, self.output_channels, 
                               (cfg.arch.cnn_dec.kernel_size + 1,
                                cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1), 
                               cfg.arch.cnn_dec.stride,
                               cfg.arch.cnn_dec.padding, bias=False),
            nn.Sigmoid()
            # state size. self.output_channels x 28 x 28
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        #print("shape deconv")
        #print(x.shape)
        #print("shape return")
        #print(x.view(-1, self.output_width*self.output_height).shape)
        return x.view(-1, self.output_width*self.output_height)

#### BMW

class BMW_CNN_Encoder(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.input_size = (cfg.arch.cnn_enc.in_channels, cfg.arch.cnn_enc.input_height, cfg.arch.cnn_enc.input_width)
        self.channel_mult = cfg.arch.cnn_enc.channel_mult

        #convolutions
        if cfg.arch.n_layers == 8:
            self.conv = nn.Sequential(
                #1
                nn.Conv2d(in_channels=cfg.arch.cnn_enc.in_channels,
                          out_channels=self.channel_mult*1,
                          kernel_size=cfg.arch.cnn_enc.kernel_size,
                          stride=cfg.arch.cnn_enc.stride,
                          padding=cfg.arch.cnn_enc.padding),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #2
                nn.Conv2d(self.channel_mult*1, self.channel_mult*2,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*2),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #3
                nn.Conv2d(self.channel_mult*2, self.channel_mult*4,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*4),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #4
                nn.Conv2d(self.channel_mult*4, self.channel_mult*8,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*8),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #5
                nn.Conv2d(self.channel_mult*8, self.channel_mult*16,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*16),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #6
                nn.Conv2d(self.channel_mult*16, self.channel_mult*32,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*32),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #7
                nn.Conv2d(self.channel_mult*32, self.channel_mult*32,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*32),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #8
                nn.Conv2d(self.channel_mult*32, self.channel_mult*32,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*32),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
            )
        elif cfg.arch.n_layers == 4:
            self.conv = nn.Sequential(
                #1
                nn.Conv2d(in_channels=cfg.arch.cnn_enc.in_channels,
                          out_channels=self.channel_mult*1,
                          kernel_size=cfg.arch.cnn_enc.kernel_size,
                          stride=cfg.arch.cnn_enc.stride,
                          padding=cfg.arch.cnn_enc.padding),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #2
                nn.Conv2d(self.channel_mult*1, self.channel_mult*2,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*2),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #3
                nn.Conv2d(self.channel_mult*2, self.channel_mult*4,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*4),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
                #4
                nn.Conv2d(self.channel_mult*4, self.channel_mult*8,
                          cfg.arch.cnn_enc.kernel_size, 2, cfg.arch.cnn_enc.padding),
                nn.BatchNorm2d(self.channel_mult*8),
                nn.LeakyReLU(0.2*cfg.arch.cnn_enc.negative_slope_factor, inplace=True),
            )


        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, cfg.arch.embedding_size),
            nn.BatchNorm1d(cfg.arch.embedding_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size).float())
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class BMW_CNN_Decoder(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.output_height = cfg.arch.cnn_dec.output_height
        self.output_width = cfg.arch.cnn_dec.output_width
        self.input_dim = cfg.arch.embedding_size
        self.channel_mult = cfg.arch.cnn_dec.channel_mult
        self.output_channels = cfg.arch.cnn_dec.out_channels
        self.fc_output_dim = cfg.arch.cnn_dec.fc_output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        if cfg.arch.n_layers == 8:
            self.deconv = nn.Sequential(
                # input is Z, going into a convolution
                #1
                nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*32,
                                   (cfg.arch.cnn_dec.kernel_size,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding,
                                   bias=False),
                nn.BatchNorm2d(self.channel_mult*32),
                nn.ReLU(True),
                #2
                nn.ConvTranspose2d(self.channel_mult*32, self.channel_mult*16,
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.channel_mult*16),
                nn.ReLU(True),
                #3
                nn.ConvTranspose2d(self.channel_mult*16, self.channel_mult*8,
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.channel_mult*8),
                nn.ReLU(True),
                #4
                nn.ConvTranspose2d(self.channel_mult*8, self.channel_mult*4,
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.channel_mult*4),
                nn.ReLU(True),
                # state size. self.channel_mult*32 x 4 x 4
                #5
                nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.channel_mult*2),
                nn.ReLU(True),
                # state size. self.channel_mult*16 x 7 x 7
                #6
                nn.ConvTranspose2d(self.channel_mult*2, self.output_channels, 
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.output_channels),
                nn.ReLU(True),
                # state size. self.channel_mult*8 x 14 x 14
                #7
                nn.ConvTranspose2d(self.output_channels, self.output_channels, 
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.output_channels),
                nn.ReLU(True),
                # state size. self.channel_mult*8 x 14 x 14
                #8
                nn.ConvTranspose2d(self.output_channels, self.output_channels, 
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                #nn.Sigmoid()
                nn.Tanh()
                # state size. self.output_channels x 28 x 28
            )
        elif cfg.arch.n_layers == 4:
            self.deconv = nn.Sequential(
                #1
                nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4,
                                   (cfg.arch.cnn_dec.kernel_size,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.channel_mult*4),
                nn.ReLU(True),
                #2
                nn.ConvTranspose2d(self.channel_mult*4, self.output_channels, 
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.output_channels),
                nn.ReLU(True),
                #3
                nn.ConvTranspose2d(self.output_channels, self.output_channels, 
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                nn.BatchNorm2d(self.output_channels),
                nn.ReLU(True),
                #4
                nn.ConvTranspose2d(self.output_channels, self.output_channels, 
                                   (cfg.arch.cnn_dec.kernel_size + 1,
                                    cfg.arch.cnn_dec.kernel_size*cfg.arch.cnn_dec.kernel_size_factor +1),
                                   cfg.arch.cnn_dec.stride,
                                   cfg.arch.cnn_dec.padding, bias=False),
                #nn.Sigmoid()
                nn.Tanh()
                # state size. self.output_channels x 28 x 28
            )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        #print("shape deconv")
        #print(x.shape)
        #print("shape return")
        #print(x.view(-1, self.output_width*self.output_height).shape)
        return x.view(-1, self.output_width*self.output_height)
