# 🪰 PestVision Challenge 🌾

Welcome to the Pest Detection Challenge! This repository houses the codebase for the competition.

## 📂 Directory Structure and Usage

- **PestVisionChallenge**: The root directory for the challenge.
    - **modelling**: Contains code for model training and evaluation.
        - `pest_detection.py`: Abstract class for pest detection, with implementations for YOLOv8. Includes methods for model loading, training, and evaluation. Extendable for other models.
    - **synthetic_data_generation**: Code for generating synthetic data.
        - `pest_blending.py`: Script for blending foreground pests on background crops to create synthetic data. Supports methods like Deep Image Blending and PCTNet image harmonization. Extendable for other techniques.
        - `save_blended_images`: Jupyter notebook demonstrating how to generate and save synthetic data.
        - `utils_deep_image_blending.py`: Utility functions for deep image blending.
        - `data_utils`:
            - `background_loader.py`: Script for loading background images. Supports Paddy disease classification and Riceleafs dataset; extendable for other datasets.
            - `foreground_loader.py`: Script for loading foreground images. Supports IP102 dataset; extendable for other datasets.
            - `foreground_mask_generation.py`: Script for generating foreground masks. Supports methods like SAM; extendable for other techniques.
            - `visualize_data.ipynb`: Jupyter notebook demonstrating how to load foreground and background datasets.
            - `save_foreground_masks.ipynb`: Jupyter notebook demonstrating how to save foreground masks.

**Note**:
- Participants must update the paths in the Jupyter notebooks to their local paths.
- Participants are encouraged to innovate in synthetic data generation and model training/fine-tuning strategies to enhance model performance on real-world data.

## 📊 Data

### Synthetic Data
Synthetic data is generated using code from the `synthetic_data_generation` directory. It combines background images from the [Paddy disease classification dataset](https://www.kaggle.com/competitions/paddy-disease-classification/data) and the [RiceLeafs dataset](https://www.kaggle.com/datasets/shayanriyaz/riceleafs) with foreground pests from the [IP102 dataset](https://github.com/xpwu95/IP102). Two blending methods are used: Deep Image Blending and PCTNet image harmonization.

#### Foreground Pests
Images from the IP102 dataset, featuring 102 classes of insects and pests.

#### Background Crops
Images from the Paddy disease classification and RiceLeafs datasets.

### Evaluation Data
Real-world images of crops infested with pests are used for model evaluation.

Note: Data details and download links will be provided soon.

## 🔧 Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/WadhwaniAI/PestVisionChallenge
    ```

2. **Install required packages**:
    ```bash
    conda env create -f environment.yml
    conda activate pestvision
    ```

## 📏 Evaluation Guidelines

The evaluation script, located in the `modelling` directory, implements the `evaluate()` method in `pest_detection.py`.

### Evaluation Metrics

- **mAP**: Standard metric for object detection.
- **Image Level Classification**: Binary assessment of pest presence.
- **Pest Count per Image**: Quantifies the severity of pest infestation.

Participants must also adhere to the following guidelines:

1. **Data Sources**: Disclose all training data sources.
2. **Methodology for Data Generation and Curation**: Provide detailed explanations of data generation and curation techniques.
3. **Model Building**: Share codebase, model architectures, and training procedures.

## 📥 Submission Guidelines

(To be updated soon)

## 🎉 Organizers

(To be updated soon)

## 🙏 Acknowledgements and Citations

(To be updated soon)


