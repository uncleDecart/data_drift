import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule

from .architectures import CNN_Encoder, CNN_Decoder, BMW_CNN_Encoder, BMW_CNN_Decoder
from sklearn.metrics import mean_squared_error

from omegaconf import DictConfig

import numpy as np

class AE(LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args
        #self.automatic_optimization=False
        if not self.args.arch.use_bmw_enc_dec:
            self.encoder = CNN_Encoder(args)
            self.decoder = CNN_Decoder(args)
        else:
            self.encoder = BMW_CNN_Encoder(args)
            self.decoder = BMW_CNN_Decoder(args)
            
        #self.train_loss = 0
       
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.args.exp.lr,
            weight_decay=1e-5,
        )
        scheduler_dict = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer
            ),
            "interval": "epoch",
            "monitor": "train_loss",
            "strict": False
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    """ # since gives warning
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    """
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(self(x), x.view(-1, self.args.arch.cnn_dec.output_height*self.args.arch.cnn_dec.output_width),
                          reduction = 'sum')
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss
    
    def evaluate(self, loader=None, save_reconstruction_every_n_batch=20):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.test_dataloader()

        reconstructions = []
        reals = []
        labels = []
        dict_mse = dict()
        batch_cnt = -1
        
        for batch in loader:
            batch_cnt = batch_cnt + 1
            print("processing the batch: {}".format(str(batch_cnt)))
            #
            X, L = batch
            if batch_cnt % save_reconstruction_every_n_batch == 0:
                reals.append(X)
                labels.append(L)
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1) #--> here we have the shape batch_size X input_width*input_height*in_channels
            reconstruction = self.forward(flattened_X) # here, the input is encoded and then decoded.
            reconstruction = reconstruction.view(X.shape[0], -1) #now we again make the shape batch_size X output_width*output_height*out_channels

            # mse per sample, real
            mse = []
            for b in range(self.args.exp.batch_size):
                mse.append(mean_squared_error(reconstruction[b].cpu().detach().numpy(),  #np.moveaxis(reconstruction[b].detach().numpy(), 0, -1)
                                              flattened_X[b].cpu().detach().numpy()))
            dict_mse[batch_cnt] = dict(zip(mse, list(labels[batch_cnt].detach().numpy())))
            #
            if batch_cnt % save_reconstruction_every_n_batch == 0:
                reconstructions.append(reconstruction.view(-1,   # here we take care of batch_size
                                                           self.args.arch.cnn_dec.out_channels,
                                                           self.args.arch.cnn_dec.output_height,
                                                           self.args.arch.cnn_dec.output_width).cpu().detach().numpy())
        return reconstructions, reals, labels, dict_mse

    def get_reconstructions(self, loader=None, per_batch_mse=False, n_batches=-1):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.val_dataloader()

        reconstructions = []
        reals = []
        labels = []
        mse_redsum = []
        mse_redmean = []
        dict_mse = dict()
        batch_cnt = -1

        for batch in loader:
            batch_cnt = batch_cnt + 1
            if n_batches < 0 or batch_cnt > n_batches:
                break
            print("processing the batch: {}".format(str(batch_cnt)))
            #
            X, L = batch
            reals.append(X)
            labels.append(L)
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1) #--> here we have the shape batch_size X input_width*input_height*in_channels
            reconstruction = self.forward(flattened_X) # here, the input is encoded and then decoded.
            reconstruction = reconstruction.view(X.shape[0], -1) #now we again make the shape batch_size X output_width*output_height*out_channels
            if not per_batch_mse:
                # mse per sample, real
                mse = []
                for b in range(self.args.exp.batch_size):
                    mse.append(mean_squared_error(reconstruction[b].cpu().detach().numpy(),  #np.moveaxis(reconstruction[b].detach().numpy(), 0, -1)
                                                  flattened_X[b].cpu().detach().numpy()))
                dict_mse[batch_cnt] = dict(zip(mse, list(labels[batch_cnt].detach().numpy())))
            else:
                mse_redsum.append(F.mse_loss(reconstruction, flattened_X, reduction = 'sum'))
                mse_redmean.append(F.mse_loss(reconstruction, flattened_X, reduction = 'mean'))
            reconstructions.append(reconstruction.view(-1,   # here we take care of batch_size
                                                       self.args.arch.cnn_dec.out_channels,
                                                       self.args.arch.cnn_dec.output_height,
                                                       self.args.arch.cnn_dec.output_width).cpu().detach().numpy())
        #return np.concatenate(reconstructions, axis=0)
        return reconstructions, reals, labels, dict_mse, mse_redsum, mse_redmean
    
