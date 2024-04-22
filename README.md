# PestVision Challenge

This repository contains the codebase for the Pest Detection Challenge. Below is the structure of the directories and files:

## Directory Structure

- **PestVisionChallenge**: Root directory for the challenge.
    - **modelling**: Contains code related to model training and evaluation.
        - `pest_detection.py`: Abstract class for pest detection, and implemented class for YOLOv8. Methods include loading model, training, and evaluation. The code is extendable for other models that participants might want to use.
    - **synthetic_data_generation**: Code for generating synthetic data.
        - `pest_blending.py`: Script for blending foreground pests on background crops to generate synthetic data. Supported blending methods include Deep Image Blending and PCTNet image harmonization from libcom library. The code is extendable for other methods that participants might want to use.
        - `save_blended_images`: Jupyter notebook demonstrating how to generate and save synthetic data.
        - `utils_deep_image_blending.py`: Utility functions for deep image blending method.
        - `data_utils`:
            - `background_loader.py`: Script for loading background images. Supported background datasets include Paddy disease classification and Riceleafs dataset; the code is extendable for other datasets.
            - `foreground_loader.py`: Script for loading foreground images. Supported foreground datasets include IP102 dataset; the code is extendable for other datasets.
            - `foreground_mask_generation.py`: Script for generating foreground masks. Supported mask generation method includes SAM; the code is extendable for other methods.
            - `visualize_data.ipynb`: Jupyter notebook demonstrating how to load foreground and background datasets.
            - `save_foreground_masks.ipynb`: Jupyter notebook demonstrating how to save foreground masks.