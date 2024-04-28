# object detection code base with abstract class

import os
from abc import ABC, abstractmethod
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


class AbstractPestDetection(ABC):
    """
    Abstract class for pest detection
    """

    def __init__(self, device):
        self.device = device

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def train(self):
        pass

    # TODO: implement evaluate method
    # @abstractmethod
    # def evaluate(self):
    #     pass


class PestDetection_yolov8(AbstractPestDetection):
    """
    YOLOv8 model for pest detection
    """

    def __init__(self, device):
        super().__init__(device)
        self.model = None

    def load_model(self, model_path):
        """
        Load the YOLOv8 model

        Parameters:

         model_path (str): path to the model checkpoint
        """

        self.model = YOLO(model_path)

    def train(
        self,
        data_yml,
        no_epochs,
        img_size,
        batch_size,
        project_path,
        exp_name,
        patience,
        pretrained,
        verbose,
        **kwargs
    ):

        results = self.model.train(
            data=data_yml,
            epochs=no_epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project=project_path,
            name=exp_name,
            pretrained=pretrained,
            verbose=verbose,
            patience=patience,
            **kwargs
        )
        self.model.val()

        return results

    # TODO: implment evaluate method
    # def evaluate(self):
    #     pass
