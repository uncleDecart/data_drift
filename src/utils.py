import os
from omegaconf import DictConfig
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

class LoadDataset():
    """ Wrapper not to load images"""
    @staticmethod
    def load_image_as_numpy(image_path: str):
        return np.asarray(Image.open(image_path).convert("RGB"))
    @staticmethod
    def load_image(image_path: str):
        return Image.open(image_path)

    def __init__(self, params, transform):
        super().__init__()

        self.img_path_list, self.label_num_map, self.label_list = params
        self.transform = transform


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        image = self.load_image_as_numpy(self.img_path_list[index])
        try:
            label = self.label_num_map[self.label_list[index]]
        except:
            print("WARNING! : {} {} {}".format(self.label_num_map, len(self.label_list), index))
        if self.transform:
            image = self.transform(image)
        return (image, label)


def save_model(trainer, cfg: DictConfig, base_path: str):
    path = os.path.join(base_path, f"{cfg.exp.results_path}/model_{cfg.exp.title}.ckpt" )
    trainer.save_checkpoint(path)

    cfg_path = os.path.join(base_path, f"{cfg.exp.title}.yaml")
    OmegaConf.save(config=cfg, f=cfg_path)
    loaded = OmegaConf.load(cfg_path)
    assert conf == loaded

#def load_model()
#    pass

def rec_err_per_label(dict_mse: dict):
    label_mse = dict()

    for v in dict_mse.values():
        for mse, l in v.items():
            label_mse.setdefault(l, []).append(mse)

    mean_error = {}

    for k, v in label_mse.items():
        mean_error[k] = np.mean(v)
    
    return mean_error

def get_representations(model, loader):
    representations = []
    labels = []

    for image in loader:
        model.eval()
        X, L = image
        labels.append(L.tolist())
        flattened_X = X.view(X.shape[0], -1)
        representations.append(model.encoder(flattened_X).data.tolist())

    return np.array(representations), np.array(labels)


def reconstruction_vis(model, loader, n_batches, batch_size):
    reconstructions, reals, _, dict_mse, _, _ = model.get_reconstructions(loader=loader, n_batches=n_batches)

    print("Original images")
    #for j in range(min(10, len(reals))):
    j=3
    for i in range(batch_size):
        plt.subplot(4, 4,i+1)
        plt.axis('off')
        plt.imshow(np.moveaxis(reals[j][i].detach().numpy(), 0, -1), interpolation='none')
    plt.tight_layout()
    plt.show()


    #for j in range(min(10, len(reconstructions))):
    #one batch
    print("Transformed")
    j=3
    for i in range(batch_size):
        plt.subplot(4, 4,i+1)
        plt.axis('off')
        plt.imshow(np.moveaxis(reconstructions[j][i], 0, -1), interpolation='none')
    plt.tight_layout()
    plt.show()

    print("Reconstruction error for labels ", rec_err_per_label(dict_mse))
