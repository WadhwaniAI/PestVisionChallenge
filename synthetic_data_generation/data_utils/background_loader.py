import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from natsort import natsorted
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

# abstract class for background crop dataset

class AbstractBackgroundDataset(Dataset, ABC):

    """
    An abstract class for foreground pest dataset.

        Parameters:

                dataset_dir (str): Path to dataset directory containing images, masks and splits.
                split (str): a split from ['train', 'val', 'test'].
                transform (callable, optional): Optional transform to be applied on a background sample.

    """

    def __init__(
        self, dataset_dir: str, split: str, transform: Optional[Callable] = None
    ):

        self.split = split
        self.dataset_dir = dataset_dir
        self.transform = transform

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

class PaddyDiseaseClassificationDataset(AbstractBackgroundDataset):

    """
    A concrete class for Paddy Disease Classification dataset.

    Parameters:
            
            dataset_dir (str): Path to dataset directory containing images, annotations, masks and splits.
                                Folder structure:
    
                                    dataset_dir
                                            ├── train: jpg files should exist here
                                            ├── val: jpg files should exist here
                                            ├── test: jpg files should exist here
    
            split (str): a split from ['train', 'val', 'test'] for which {split}.txt file exists.
            transform (callable, optional): Optional transform to be applied on a background sample.
            target_size (int): Target size for the image.
    """
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        target_size: int = 512,
    ):
    
        super().__init__(dataset_dir, split, transform)

        self.ts = target_size

        self.images = natsorted(
            glob(os.path.join(self.dataset_dir, split, "*", "*.jpg"))
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        """
        Returns a sample from the dataset at the given index.
        
        Parameters:
            idx (int): Index of the sample to be returned.

        Returns:
            Dict[str, Any]: A sample from the dataset at the given index.
        """

        img_file = self.images[idx]
        image = Image.open(img_file).convert("RGB").resize((self.ts, self.ts))

        sample = {"image": image, "image_filename": img_file}

        if self.transform:
            sample = self.transform(sample, self.ts)

        return sample

class RiceLeafsDataset(AbstractBackgroundDataset):

    """
    A concrete class for Paddy Disease Classification dataset.

    Parameters:
            
            dataset_dir (str): Path to dataset directory containing images, annotations, masks and splits.
                                Folder structure:
    
                                    dataset_dir
                                            ├── train: jpg files should exist here
                                            ├── val: jpg files should exist here
                                            ├── test: jpg files should exist here
    
            split (str): a split from ['train', 'val', 'test'] for which {split}.txt file exists.
            transform (callable, optional): Optional transform to be applied on a background sample.
            target_size (int): Target size for the image.
    """
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        target_size: int = 512,
    ):
    
        super().__init__(dataset_dir, split, transform)

        self.ts = target_size

        self.images = natsorted(
            glob(os.path.join(self.dataset_dir, split, "*", "*.jpg"))
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        """
        Returns a sample from the dataset at the given index.
        
        Parameters:
            idx (int): Index of the sample to be returned.

        Returns:
            Dict[str, Any]: A sample from the dataset at the given index.
        """

        img_file = self.images[idx]
        image = Image.open(img_file).convert("RGB").resize((self.ts, self.ts))

        sample = {"image": image, "image_filename": img_file}

        if self.transform:
            sample = self.transform(sample, self.ts)

        return sample

class BackgroundRandomCrop(object):
    """
    A class to apply random crop on background crop images.

    """

    def __init__(self, crop_prob: float = 0.6):

        """
        Parameters:
            crop_prob (float): Probability of applying random crop.
        """

        self.crop_prob = crop_prob

    def __call__(self, sample: Dict[str, Any], ts: int) -> Dict[str, Any]:

        """
        Applies random crop on the background image.

        Parameters:

            sample (Dict[str, Any]): A sample from the dataset.

        Returns:
            
                Dict[str, Any]: Transformed sample.
        """

        if torch.rand(1).item() < self.crop_prob:

            resize_cropper = T.RandomResizedCrop(size=(ts, ts))
            sample["image"] = resize_cropper(sample["image"])

        return sample


