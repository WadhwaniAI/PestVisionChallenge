import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from natsort import natsorted
from glob import glob
import xml.etree.ElementTree as ET
from PIL import Image 
import numpy as np
import shutil
import cv2
from tqdm import tqdm
from typing import List, Tuple, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

# abstract class for mask generation, 
# and a concrete class for generating masks using the SAM model.

class AbstractMaskGenerator(ABC):
    """
    An abstract class for mask generation.
    """
    
    def __init__(self, model_type: str, checkpoint_path: str, device: str):
        """
        Initialize the mask generator.
        
        Args:
            model_type (str): model type.
            checkpoint_path (str): path to the checkpoint.
            device (str): device to run the model on.
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        

    @abstractmethod
    def generate_mask(self, img: str, ann: str) -> None:
        """
        Abstract method to generate mask for an image.
        
        Args:
            img (str): path to the image file.
            ann (str): path to the annotation file.
            mask_save_path (str): path to save the generated mask.
        """
        pass

class SAM_MaskGenerator(AbstractMaskGenerator):
    """
    A class to generate masks using the SAM model.
    """

    def __init__(self, model_type: str, checkpoint_path: str, device: str):
        """
        Initialize the SAM mask generator.

            Parameters:
                model_type (str): model type.
                checkpoint_path (str): path to the checkpoint.
                device (str): device to run the model on.
        """
        super().__init__(model_type, checkpoint_path, device)

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
        self.mask_predictor = SamPredictor(self.sam)

    def generate_mask(self, img: str, ann: str) -> Image:

        tree = ET.parse(ann) 
        root = tree.getroot()
        bbox_coordinates = []

        for member in root.findall('object'):
            class_name = member[0].text
            
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            
            bbox_coordinates.append([xmin, ymin, xmax, ymax])

        # saving the mask for only the first pest in the image in case of multiple pests!
        box = np.array(bbox_coordinates)[0]

        image_bgr = cv2.imread(img)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self.mask_predictor.set_image(image_rgb)

        masks, scores, logits = self.mask_predictor.predict(
            box=box,
            multimask_output=True
        )
        
        data = Image.fromarray(masks[0])
        return data

