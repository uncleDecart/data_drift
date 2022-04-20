from PIL import Image
import pytorch_lightning as pl
from .utils import LoadDataset
from typing import Callable, Optional, List
from torch.utils.data import DataLoader  # type: ignore
import random
import numpy as np

class BMWDataset(pl.LightningDataModule):
    def __init__(self,
            df,
            split: float = 0.5,
            transforms: Optional[Callable] = None,
            drop_last: bool = False,
            batch_size: int = 10,
            num_workers: int = 0
    ):
        super().__init__()
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.transforms = transforms

        good_lbls = df[df.lbl == 'IO']
        bad_lbls = df[df.lbl == 'NIO']

        label_num_map = {'IO': 0, 'NIO': 1}

        train_len = int(len(good_lbls)*split)
        train_idx = np.random.choice([i for i in range(len(good_lbls))], train_len, replace=False)
        train_objs = list(zip(good_lbls.iloc[train_idx].file_name.tolist(), ['IO']*len(train_idx)))
        random.shuffle(train_objs)
        train_paths, train_lbls = zip(*train_objs)
        self.__trainset = LoadDataset((train_paths, label_num_map, train_lbls), self.transforms)

        test_idx = np.array(list((set(range(len(good_lbls))) - set(train_idx))))
        test_img_path = good_lbls.iloc[test_idx].file_name.tolist() + bad_lbls.file_name.tolist()
        test_labels = ['IO']*len(test_idx) + ['NIO']*len(bad_lbls)
        test_objs = list(zip(test_img_path, test_labels))
        random.shuffle(test_objs)
        test_paths, test_lbls = zip(*test_objs)
        self.__testset = LoadDataset((test_paths, label_num_map, test_lbls), self.transforms)

        self.__trainloader: DataLoader = DataLoader(self.__trainset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers,
                                                    drop_last=self.drop_last)
        self.__testloader: DataLoader = DataLoader(self.__testset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   drop_last=self.drop_last)
        
    def train_dataloader(self):
        return self.__trainloader

    def train_dataset(self):
        return self.__trainset

    def test_dataloader(self):
        return self.__testloader

    def test_dataset(self):
        return self.__testset
