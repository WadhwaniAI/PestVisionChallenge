# PestVision Challenge

This repository contains the codebase for the Pest Detection Challenge. 

## Directory Structure and usage

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

Participants are encouraged to think creatively, experiment, and innovate in synthetic data generation and/or model training/fine-tuning strategies to boost their models' performance on a hidden real-world test set.

## Data

### Synthetic Data
The synthetic data is generated using the code in the `synthetic_data_generation` directory. It comprises background images sourced from the [Paddy disease classification dataset](!https://www.kaggle.com/competitions/paddy-disease-classification/data) and [RiceLeafs dataset](!https://www.kaggle.com/datasets/shayanriyaz/riceleafs), along with foreground pests from the [IP102 dataset](!https://github.com/xpwu95/IP102). The synthesis process involves blending the foreground pests onto the background crops using [Deep Image Blending](!https://github.com/owenzlz/DeepImageBlending) and [PCTNet image harmonization from libcom](!https://libcom.readthedocs.io/en/latest/image_harmonization.html) methods. 

#### Foreground Pests
The foreground pests consist of images sourced from the , which encompasses 102 classes of insects and pests. 

#### Background Crops
The background crops are images sourced from the Paddy disease classification dataset and the RiceLeafs dataset. 

### Evaluation Data
The evaluation data comprises real-world images of crops infested with pests. These images serve as the basis for evaluating the performance of pest detection models in real-world scenarios.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/WadhwaniAI/PestVisionChallenge     
    ```

2. Install the required packages:
    ```bash
    conda env create -f environment.yml
    conda activate pestvision
    ```

## Evaluation Guidelines

The evaluation script will be updated in the `modelling` directory, implemented as the `evaluate()` method in `pest_detection.py`.

### Evaluation Metrics

- **mAP**: This standard metric for object detection quantifies the model's accuracy. A higher mAP score indicates more accurate bounding box predictions.
- **Image Level Classification**: This metric evaluates whether pests are present in individual images, providing a binary assessment of pest presence.
- **Pest Count per Image**: This metric evaluates the model's accuracy in detecting the number of pests in each image, crucial for assessing the severity of pest infestation.

In addition to these metrics, participants are required to adhere to certain guidelines:

1. **Data Sources**: Participants must disclose the sources of their training data, including any synthetic data generated, external datasets utilized, and proprietary datasets incorporated.
2. **Methodology for Data Generation and Curation**: Participants should provide detailed explanations of the techniques and processes used to generate and curate their training data, including any augmentation methods applied.
3. **Model Building**: Participants are required to share their codebase, including model architectures, training procedures, and any fine-tuning techniques utilized to enhance model performance.

## Submission Guidelines

(To be updated soon)

## Organizers

(To be updated soon)

## Acknowledgements and citations

- Paddy disease classification dataset, RiceLeafs dataset, and IP102 dataset are publicly available datasets. The organizers are grateful to the authors for making these datasets available for research purposes.

```
@inproceedings{zhang2020deep,
  title={Deep Image Blending},
  author={Zhang, Lingzhi and Wen, Tarmily and Shi, Jianbo},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={231--240},
  year={2020}
}
```

```
@article{niu2021making,
  title={Making images real again: A comprehensive survey on deep image composition},
  author={Niu, Li and Cong, Wenyan and Liu, Liu and Hong, Yan and Zhang, Bo and Liang, Jing and Zhang, Liqing},
  journal={arXiv preprint arXiv:2106.14490},
  year={2021}
}
```

```
@InProceedings{Guerreiro_2023_CVPR,
    author    = {Guerreiro, Julian Jorge Andrade and Nakazawa, Mitsuru and Stenger, Bj\"orn},
    title     = {PCT-Net: Full Resolution Image Harmonization Using Pixel-Wise Color Transformations},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5917-5926}
}
```

```
@inproceedings{Wu2019Insect,
  title={IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition},
  author={Xiaoping Wu and Chi Zhan and Yukun Lai and Ming-Ming Cheng and Jufeng Yang},
  booktitle={IEEE CVPR},
  pages={8787--8796},
  year={2019},
}
```
