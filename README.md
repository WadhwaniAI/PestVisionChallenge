# 🪰 PestVision Challenge 🌾

Welcome to the PestVision Challenge! This repository houses the codebase for the competition. Please check the [challenge website](https://pestvision.wadhwaniai.org/) for more details and registration process

## 📂 Directory Structure and Usage

- **PestVisionChallenge**: The root directory for the challenge.
    - **modelling**: Contains code for model training and evaluation.
        - `pest_detection.py`: Abstract class for pest detection, with implementations for YOLOv8. Includes methods for model loading, training, and evaluation. Extendable for other models.
    - **synthetic_data_generation**: Code for generating synthetic data.
        - `pest_blending.py`: Script for blending foreground pests on background crops to create synthetic data. Supports  two methods, namely Deep Image Blending and PCTNet image harmonization. Extendable for other techniques.
        - `save_blended_images`: Jupyter notebook demonstrating how to generate and save synthetic data.
        - `utils_deep_image_blending.py`: Utility functions for deep image blending.
        - `data_utils`:
            - `background_loader.py`: Script for loading background images. Supports Paddy disease classification and Riceleafs dataset; extendable for other datasets.
            - `foreground_loader.py`: Script for loading foreground images. Supports IP102 dataset; extendable for other datasets.
            - `foreground_mask_generation.py`: Script for generating foreground masks. Supports methods like SAM; extendable for other techniques.
            - `visualize_data.ipynb`: Jupyter notebook demonstrating how to load foreground and background datasets.
            - `save_foreground_masks.ipynb`: Jupyter notebook demonstrating how to save foreground masks.
            - `change_class_ids.ipynb`: Utility script for converting class label IDs into YOLO format, essential for aggregating all pest classes into a single class, facilitating class-agnostic pest detection

**Note**:
- Participants must update the paths in the Jupyter notebooks to their local paths.
- Participants are encouraged to innovate in synthetic data generation and model training/fine-tuning strategies to enhance model performance on evaluation data.

## 📊 Data

### Synthetic Data
Synthetic data is generated using code from the `synthetic_data_generation` directory. It combines background images from the [Paddy disease classification dataset](https://www.kaggle.com/competitions/paddy-disease-classification/data) and the [Rice Leafs dataset](https://www.kaggle.com/datasets/shayanriyaz/riceleafs) with foreground pests from the [IP102 dataset](https://github.com/xpwu95/IP102). Two blending methods are used: [Deep Image Blending](https://github.com/owenzlz/DeepImageBlending) and [PCTNet image harmonization](https://libcom.readthedocs.io/en/latest/image_harmonization.html).

#### Foreground Pests
Images from the IP102 dataset, featuring 102 classes of insects and pests.

#### Background Crops
Images from the Paddy disease classification and Rice Leafs datasets.

### Evaluation Data
Real-world images of crops infested with pests are used for model evaluation. 

**Note**: Find more details [here](https://pestvision.wadhwaniai.org/data/index.html) (data will be available on registration to the challenge). 

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

4. **Download checkpoint for ViT-H SAM model**:

    Only required if using the ViT-H SAM model for foreground mask generation (in `foreground_mask_generation.py`). 

    ```bash
    cd synthetic_data_generation/data_utils
    mkdir weights
    cd weights
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

## 📏 📥 Evaluation and Submission Guidelines

The evaluation script will be updated in the `modeling` directory, implemented as `evaluate()` method in `pest_detection.py`.

### Evaluation Metrics

- **mAP**: Standard metric for object detection.
- **Image Level Classification**: Binary assessment of pest presence.
- **Pest Count per Image**: Quantifies the severity of pest infestation.

Participants must also adhere to the following guidelines:

1. **Data Sources**: Disclose all training data sources.
2. **Methodology for Data Generation and Curation**: Provide detailed explanations of data generation and curation techniques.
3. **Model Building**: Share codebase, model architectures, and training procedures.

Find more details about evaluation and submission [here](https://pestvision.wadhwaniai.org/submission/index.html).


## 🚀 Organizers

This challenge is organized by [Wadhwani AI](https://www.wadhwaniai.org/), a non-profit organization dedicated to leveraging artificial intelligence for social impact. For any queries, please contact the PestVision team at pestvision@wadhwaniai.org.

## 🙏 Acknowledgements and Citations

### Data:

The organizers extend their sincere gratitude to the MS Swaminathan Research Foundation ([MSSRF](https://www.mssrf.org/)) for generously granting permission to include some of their collected data in the evaluation set for this competition.

IP102 dataset, Paddy disease classification dataset, Rice Leafs dataset and the AgriPest dataset are publicly available datasets. The organizers are grateful to the authors for making these datasets available for research purposes.

- **IP102 dataset**: Wu, X., Zhan, C., Lai, Y.K., Cheng, M.M. and Yang, J., 2019. Ip102: A large-scale benchmark dataset for insect pest recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8787-8796).
- **Paddy disease classification dataset**: Petchiammal A, Briskline Kiruba S, Murugan D, Pandarasamy Arjunan, November 18, 2022, "Paddy Doctor: A Visual Image Dataset for Automated Paddy Disease Classification and Benchmarking", IEEE Dataport, doi: https://dx.doi.org/10.21227/hz4v-af08.
- **Rice Leafs dataset**: Shayan Riyaz, "Riceleafs", Kaggle, https://www.kaggle.com/datasets/shayanriyaz/riceleafs
- **AgriPest dataset**: Wang, R., Liu, L., Xie, C., Yang, P., Li, R. and Zhou, M., 2021. Agripest: A large-scale domain-specific benchmark dataset for practical agricultural pest detection in the wild. Sensors, 21(5), p.1601.

### Code:

- **Deep Image Blending** (https://github.com/owenzlz/DeepImageBlending):
Zhang, L., Wen, T. and Shi, J., 2020. Deep image blending. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (pp. 231-240).
- **PCTNet from libcom** (https://github.com/bcmi/libcom): Niu, L., Cong, W., Liu, L., Hong, Y., Zhang, B., Liang, J. and Zhang, L., 2021. Making images real again: A comprehensive survey on deep image composition. arXiv preprint arXiv:2106.14490.
- **Segment Anything** (https://github.com/facebookresearch/segment-anything): Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y. and Dollár, P., 2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4015-4026).


