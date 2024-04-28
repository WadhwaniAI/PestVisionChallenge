# object detection code base with abstract class

import os
from abc import ABC, abstractmethod
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


class AbstractPestDetection(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def load_model(self, model_name):
        pass

    @abstractmethod
    def train(self):
        pass

    # TODO: implement evaluate method
    # @abstractmethod
    # def evaluate(self):
    #     pass


class PestDetection_yolov8(AbstractPestDetection):

    def __init__(self, device):
        super().__init__(device)
        self.model = None

    def load_model(self, model_name):
        self.model = YOLO(f"{model_name}.pt")

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
        )
        self.model.val()

        return results

    # TODO: implment evaluate method
    # def evaluate(self):
    #     pass
