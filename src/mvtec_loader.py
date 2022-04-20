from PIL import Image
import glob
import os
from typing import Callable, Optional, List
from torch.utils.data import DataLoader  # type: ignore
from typing import Tuple
import numpy as np
import random
import pytorch_lightning as pl
import pickle
from .utils import LoadDataset


class MVTecADDataset(pl.LightningDataModule):
    """`MVTecAD <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_ Dataset.

    Args:
        data_root (str): path to data directory contains MVTecAD
        results_root (str): path to store data split
        idx_file(str): path to load previously generated data split
        targets_file(str): path to load previously created target
        targets(list): target object(s). e.g. bottle, cable, capsule.
        create_or_load(bool): if true we're creating data split for targets given, else load it
        transforms(Optional[Callable], optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.ToTensor``
        img_ext(str, optional): image file extension that are to be loaded. Defaults to ".png".
        reprod(bool): if True dump split into pickle file
    Raises:
        FileNotFoundError: [description]
    """

    def __init__(self,
        data_root: str,
        results_root: str,
        idx_file: None,  
        targets_file: None,
        targets: list,
        create_or_load: bool = True, # True --> create, False --> load
        transforms: Optional[Callable] = None,
        img_ext: str = ".png",
        val_ratio = 0.0,
        reprod: bool = False,
        drop_last: bool = True,
        batch_size: int = 10,
        num_workers: int = 0,         
    ):
        super().__init__()
        if not create_or_load:
            assert idx_file is not None
            assert os.path.isfile(os.path.join(results_root, idx_file))

        self.data_root = data_root
        self.results_root = results_root
        self.transforms = transforms
        self.targets = targets
        self.create_or_load = create_or_load
        self.reprod = reprod
        self.val_ratio = val_ratio
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.img_ext = img_ext
        self.max_labels_idx = 0

        self.idx_file = idx_file
        self.targets_file = targets_file
        if self.reprod:
            assert self.idx_file is not None
            assert self.targets_file is not None
            if not os.path.isdir(self.results_root):
                os.mkdir(self.results_root)

        self._train_idx = None
        self._val_idx = None
        
        if self.create_or_load:
            target_list = self._get_target_list()
        else: 
            #load 
            try:
                with open(os.path.join(self.results_root, self.targets_file), "rb") as f:
                    target_list = pickle.load(f)
            except IOError:
                print("File not accessible")

        if len(set(target_list).intersection(set(targets))) == 0:
            raise FileNotFoundError("none of targets found in `{data_root}'")
        self.targets = list(set(targets).intersection(target_list))
        # logger would be better but let's jus print
        print("targets to be used are: ", ','.join(self.targets))

        if self.create_or_load:
            self.__trainset = LoadDataset(self._create_loader('train'),
                                          self.transforms)

            self.test_img_list, label_num_map, label_list = self._create_loader("test")
            self.__testset = LoadDataset((self.test_img_list, label_num_map, label_list),
                                         self.transforms)

            if self.val_ratio > 0.0: 
                self.__valset = LoadDataset(self._create_loader("val"),
                                            self.transforms)
        else:
            #only train and test for now
            self.__trainset = LoadDataset(self._load_existing_loader('train', idx_file=self.idx_file),
                                          self.transforms)
            self.__testset = LoadDataset(self._load_existing_loader('test', idx_file=self.idx_file),
                                          self.transforms)


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
        if self.val_ratio > 0.0:    
            self.__valloader: DataLoader = DataLoader(self.__valset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      drop_last=self.drop_last)

    def _create_loader(self, split):
        # workaround since mvtec-ad data does not provide validation folder
        if split == 'val':
            split = 'train'

        img_path_list = []
        label_list = []
        label_num_map = dict()

        for trg in self.targets:
            labels = []
            labels = self._get_labels(os.path.join(self.data_root, trg, split))
            labels_num = list(np.array(range(len(labels))) + self.max_labels_idx)
            self.max_labels_idx = max(labels_num)
            label_num_map.update(dict(zip(labels, labels_num)))

            img_path_list = img_path_list + glob.glob(os.path.join(self.data_root, trg, split, f"*/*{self.img_ext}"))
            label_list = label_list + [self._get_label_from_path(path) for path in img_path_list]
            
        # workaround since mvtec-ad data does not provide validation folder
        if split != 'test': # 'train' or 'val'
            idx = list(range(len(img_path_list)))
            random.shuffle(idx)
            self._train_idx = idx
            if self._train_idx is None and self._val_idx is None: # do it once
                self._train_idx = idx[0:int(len(idx)*(1-self.val_ratio))]
                self._val_idx = idx[int(len(idx)*(1-self.val_ratio)):]
        # save
        if self.reprod:
            with open(os.path.join(self.results_root, self.idx_file + '_split') +".pkl", "wb") as f:
                pickle.dump(self._train_idx, f)        
            with open(os.path.join(self.results_root, self.targets_file) +".pkl", "wb") as f:
                pickle.dump(self.targets, f)        
                
        # workaround again
        if split == 'train':
            return [img_path_list[ii] for ii in self._train_idx], label_num_map, [label_list[ii] for ii in self._train_idx]
        elif split == 'val':
            return [img_path_list[ii] for ii in self._val_idx], label_num_map, [label_list[ii] for ii in self._val_idx]
        else:
            return img_path_list, label_num_map, label_list

    #only train and test 
    def _load_existing_loader(self, split, idx_file=None):
        assert idx_file is not None
        #
        img_path_list = []
        label_list = []
        label_num_map = dict()
        #
        for trg in self.targets:
            labels = []
            labels = self._get_labels(os.path.join(self.data_root, trg, split))
            label_num_map.update(dict(zip(labels, range(len(labels)))))
            #
            img_path_list = img_path_list + glob.glob(os.path.join(self.data_root, trg, split, f"*/*{self.img_ext}"))
            label_list = label_list + [self._get_label_from_path(path) for path in img_path_list]

        if split == 'train':
            try:
                with open(os.path.join(self.results_root, idx_file), "rb") as f:
                    self._train_idx = pickle.load(f)
            except IOError:
                print("File not accessible")
            #    
            return [img_path_list[ii] for ii in self._train_idx], label_num_map, [label_list[ii] for ii in self._train_idx]
        else:
            return img_path_list, label_num_map, label_list
        
    def train_idx(self):
        return self._train_idx
        
    def train_dataloader(self):
        return self.__trainloader

    def train_dataset(self):
        return self.__trainset

    def test_dataloader(self):
        return self.__testloader
    """#since no validation data is produced
    def val_dataloader(self):
        return self.__valloader
    """
    def test_dataset(self):
        return self.__testset

    def _get_target_list(self) -> List[str]:
        return os.listdir(self.data_root)

    def _get_labels(self, target_root: str) -> List[str]:
        labels = os.listdir(target_root)
        # append `good' at beggning of class list to give it label 0
        labels.pop(labels.index("good"))
        labels = ["good"] + labels
        return labels

    def _get_label_from_path(self, img_path: str) -> str:
        img_path_elems = img_path.split(os.sep)
        return img_path_elems[-2]

