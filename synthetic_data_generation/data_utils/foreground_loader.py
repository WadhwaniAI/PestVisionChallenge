import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from natsort import natsorted
from PIL import Image
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

# Abstract class for foreground pest dataset


class AbstractForegroundPestDataset(Dataset, ABC):
    """
    An abstract class for foreground pest dataset.

    """

    def __init__(
        self, dataset_dir: str, split: str, transform: Optional[Callable] = None
    ):
        """
        Parameters:

                dataset_dir (str): Path to dataset directory containing images, masks and splits.
                split (str): a split from ['train', 'val', 'test'].

        """

        self.split = split
        self.dataset_dir = dataset_dir
        self.transform = transform

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass


class IP102Dataset(AbstractForegroundPestDataset):
    """
    A concrete class for IP102 dataset.

    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        source_image_range_big: Tuple[int, int],
        source_image_range_small: Tuple[int, int],
        transform: Optional[Callable] = None,
    ):
        """
        Parameters:

        dataset_dir (str): Path to dataset directory containing images, annotations, masks and splits.
                            Folder structure:

                                dataset_dir
                                        ├── Annotations: xml files should exist here
                                        ├── ImageSets: {train, val, test}.txt files should exist here
                                        ├── JPEGImages: jpg files should exist here
                                        ├── Masks: mask files (.png) should exist here

        split (str): a split from ['train', 'val', 'test'] for which {split}.txt file exists.
        source_range_big ([int, int]): source (foreground) size range for big pests.
        source_range_small ([int, int]): source(foreground) size range for small pests.
        """

        super().__init__(dataset_dir, split, transform)

        self.source_image_range_big = source_image_range_big
        self.source_image_range_small = source_image_range_small

        images_dir = os.path.join(dataset_dir, "JPEGImages")
        annotations_dir = os.path.join(dataset_dir, "Annotations")
        masks_dir = os.path.join(dataset_dir, "Masks")
        split_txt = os.path.join(dataset_dir, f"ImageSets/{split}.txt")

        with open(split_txt, "r") as f:

            file_names = [os.path.splitext(line.strip())[0] for line in f]

        self.images = [os.path.join(images_dir, f"{name}.jpg") for name in file_names]
        self.annotations = [
            os.path.join(annotations_dir, f"{name}.xml") for name in file_names
        ]
        self.masks = [os.path.join(masks_dir, f"{name}.png") for name in file_names]

        assert (
            len(self.images) == len(self.annotations) == len(self.masks)
        ), "AssertionError: len(self.images) == len(self.annotations) == len(self.masks) should be True"

    def __len__(self) -> int:
        """Returns the length of the dataset."""

        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample from the dataset.

            Parameters:
                idx (int): index of the sample to return.

            Returns:
                Dict[str, Any]: A sample from the dataset.
        """

        mask_file = self.masks[idx]
        source_file = self.images[idx]
        ann_file = self.annotations[idx]

        mask_img_orig = Image.open(mask_file).convert("L")
        mask_img_temp = np.array(mask_img_orig)
        mask_img_temp[mask_img_temp > 0] = 1

        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()  # get root object
        except:
            print(f"error in {ann_file}")

        # bbox_coordinates = []
        class_names = []
        for member in root.findall("object"):
            class_name = member[0].text  # class name

            # bbox coordinates
            # xmin = int(member[4][0].text)
            # ymin = int(member[4][1].text)
            # xmax = int(member[4][2].text)
            # ymax = int(member[4][3].text)
            # # store data in list
            # bbox_coordinates.append([xmin, ymin, xmax, ymax])
            class_names.append(class_name)

        set_class_names = set(class_names)

        if len(set_class_names) > 1:
            print("check the dataset: len(set_class_names) > 1")
        else:
            class_name = int(class_names[0])

        ratio_filled = np.sum(mask_img_temp) / (
            mask_img_temp.shape[0] * mask_img_temp.shape[1]
        )

        if ratio_filled > 0.15:  # big insect
            is_big = True
            ss = np.random.randint(
                low=self.source_image_range_big[0], high=self.source_image_range_big[1]
            )
            mask_img_resized = mask_img_orig.resize((ss, ss))
            source_img_resized = Image.open(source_file).convert("RGB").resize((ss, ss))

        else:  # small insect
            is_big = False
            ss = np.random.randint(
                low=self.source_image_range_small[0],
                high=self.source_image_range_small[1],
            )
            mask_img_resized = mask_img_orig.resize((ss, ss))
            source_img_resized = Image.open(source_file).convert("RGB").resize((ss, ss))

        sample = {
            "source_img_resized": source_img_resized,
            "mask_img_resized": mask_img_resized,
            "pest_class_id": class_name,
            "source_size": ss,
            "source_filename": source_file,
            "is_big": is_big,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# custom transforms for foreground pest dataset:


class ForegroundBlur(object):
    """
    A class to apply random blur on foreground pest images.

    """

    def __init__(self, blur_prob: float = 0.35):
        """
        Parameters:
            blur_prob (float): probability of applying blur.
        """

        self.blur_prob = blur_prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies random blur on the source image.

        Parameters:
            sample (Dict[str, Any]): A sample from the dataset.

        Returns:
            Dict[str, Any]: Transformed sample.
        """

        if torch.rand(1).item() < self.blur_prob:

            blurrer = T.GaussianBlur(kernel_size=(5, 7), sigma=(0.1, 3))
            sample["source_img_resized"] = blurrer(sample["source_img_resized"])

        return sample


class ForegroundRotate(object):
    """
    A class to apply random rotation on foreground pest images.

    """

    def __init__(self, rotation_prob: float = 0.35):
        """
        Parameters:
            rotation_prob (float): probability of applying rotation.
        """

        self.rotation_prob = rotation_prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies random blur on the source image.

        Parameters:
            sample (Dict[str, Any]): A sample from the dataset.

        Returns:
            Dict[str, Any]: Transformed sample.
        """

        if torch.rand(1).item() < self.rotation_prob:

            angle = torch.randint(-180, 180, (1,)).item()
            sample["source_img_resized"] = T.functional.rotate(
                img=sample["source_img_resized"], angle=angle
            )
            sample["mask_img_resized"] = T.functional.rotate(
                img=sample["mask_img_resized"], angle=angle
            )

        return sample


if __name__ == "__main__":

    dataset_dir = "/bucket/siddhi/pestvision_data/foreground/Detection_IP102/VOC2007"

    transform_BlurRotate = T.Compose(
        [ForegroundBlur(blur_prob=0.35), ForegroundRotate(rotation_prob=0.35)]
    )

    ip102_dataset = IP102Dataset(
        dataset_dir=dataset_dir,
        split="train",
        source_image_range_big=(100, 200),
        source_image_range_small=(50, 100),
        transform=transform_BlurRotate,
    )

    print(len(ip102_dataset))
