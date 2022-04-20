import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule

from .architectures import CNN_Encoder, CNN_Decoder

from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error


class VAE(LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args

        self.encoder = CNN_Encoder(self.args)
        self.decoder = CNN_Decoder(self.args)

        self.var = nn.Linear(self.args.arch.cnn_enc.output_size, self.args.arch.embedding_size)
        self.mu = nn.Linear(self.args.arch.cnn_enc.output_size, self.args.arch.embedding_size)

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
        x = self.encoder(x)
        mu, logvar = self.mu(x), self.var(x)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        
        return self.decoder(z), mu, logvar

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
            reconstruction, mu, logvar = self.forward(flattened_X) # here, the input is encoded and then decoded.
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
            #
            reconstructions.append(reconstruction.view(-1,   # here we take care of batch_size
                                                       self.args.arch.cnn_dec.out_channels,
                                                       self.args.arch.cnn_dec.output_height,
                                                       self.args.arch.cnn_dec.output_width).cpu().detach().numpy())
        #return np.concatenate(reconstructions, axis=0)
        return reconstructions, reals, labels, dict_mse, mse_redsum, mse_redmean


    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

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
        x_hat, mu, logvar = self(x)

        BCE = F.binary_cross_entropy(x_hat,
                x.view(-1, self.args.arch.cnn_dec.output_height*self.args.arch.cnn_dec.output_width), reduction='sum')
        #loss = F.mse_loss(x_hat, x.view(-1, self.args.arch.cnn_dec.output_height*self.args.arch.cnn_dec.output_width), reduction = 'sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KDE = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KDE
        print(loss)
        self.log(f"{stage}_loss", loss, on_step=True)
        #self.log(f"{stage}_mu", mu, on_step=True)
        #self.log(f"{stage}_logvar", logvar, on_step=True)
        return loss
