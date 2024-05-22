from torch import BoolTensor, IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import csv
import os
import os
import csv
from PIL import Image
from typing import List

# pest detection evaluation metrics:
# 1. mean average precision (mAP) for object detection
# 2. image level classification accuracy (pest present or not)
# 3. mean absolute error for pest count per image


def calculate_mAP(csv_mapping_file: str, predictions_dir: str) -> float:
    """
    Calculate mean average precision (mAP) for object detection task

    Parameters:
        csv_mapping_file: path to csv mapping file (interal use only: not for participants)
        predictions_dir: path to directory containing prediction files (submitted by participants)

    Returns:
        mAP: mean average precision

    """

    # load annotations and predictions from csv mapping file, and make lists as required by torchmetrics

    annotations_list = []
    predictions_list = []

    with open(csv_mapping_file) as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:

            local_image_path = row[0]
            new_image_path = row[1]
            new_annotation_path = row[2]

            # base_name from new_image_path
            base_name = os.path.basename(new_image_path).split('.')[0]

            image = Image.open(new_image_path)
            image_width, image_height = image.size

            # load annotations

            boxes = []
            labels = []

            if not os.path.exists(new_annotation_path):
                annotations_list.append(
                    {'boxes': Tensor([]), 'labels': IntTensor([])})

            else:

                with open(new_annotation_path) as f:

                    lines = f.readlines()

                    if len(lines) == 0:
                        annotations_list.append(
                            {'boxes': Tensor([]), 'labels': IntTensor([])})

                    else:

                        for line in lines:
                            class_id, x_center, y_center, yolo_width, yolo_height = map(
                                float, line.split())
                            class_id = int(class_id)

                            left = (x_center - yolo_width/2) * image_width
                            top = (y_center - yolo_height/2) * image_height
                            right = (x_center + yolo_width/2) * image_width
                            bottom = (y_center + yolo_height/2) * image_height

                            boxes.append([left, top, right, bottom])
                            labels.append(class_id)

                        annotations_list.append(
                            {'boxes': Tensor(boxes), 'labels': IntTensor(labels)})

            # load predictions

            boxes = []
            scores = []
            labels = []

            prediction_file_path = os.path.join(
                predictions_dir, base_name + '.txt')

            if not os.path.exists(prediction_file_path):
                predictions_list.append(
                    {'boxes': Tensor([]), 'scores': Tensor([]), 'labels': IntTensor([])})

            else:

                with open(prediction_file_path) as f:

                    lines = f.readlines()

                    if len(lines) == 0:

                        predictions_list.append(
                            {'boxes': Tensor([]), 'scores': Tensor([]), 'labels': IntTensor([])})

                    else:

                        for line in lines:

                            class_id, x_center, y_center, yolo_width, yolo_height, score = map(
                                float, line.split())

                            class_id = int(class_id)

                            left = (x_center - yolo_width/2) * image_width
                            top = (y_center - yolo_height/2) * image_height
                            right = (x_center + yolo_width/2) * image_width
                            bottom = (y_center + yolo_height/2) * image_height

                            boxes.append([left, top, right, bottom])
                            labels.append(class_id)
                            scores.append(score)

                        predictions_list.append(
                            {'boxes': Tensor(boxes), 'scores': Tensor(scores), 'labels': IntTensor(labels)})

    # Initialize metric
    metric = MeanAveragePrecision(iou_type="bbox")

    # Update metric with predictions and respective ground truth
    metric.update(predictions_list, annotations_list)

    # Compute the results
    result = metric.compute()

    return result['map']


def calculate_image_level_classification(csv_mapping_file: str, predictions_dir: str) -> float:
    """
    Calculate image level classification accuracy (pest present or not)

    Parameters:
        csv_mapping_file: path to csv mapping file (interal use only: not for participants)
        predictions_dir: path to directory containing prediction files (submitted by participants)

    Returns:
        accuracy: image level classification accuracy
    """

    # first list for actual labels (0 or 1) and second list for predicted labels (0 or 1)
    # here, 0 = no pest, 1 = pest present (any class or number of pests)

    actual_labels = []
    predicted_labels = []

    with open(csv_mapping_file) as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:

            local_image_path = row[0]
            new_image_path = row[1]
            new_annotation_path = row[2]

            # base_name from new_image_path
            base_name = os.path.basename(new_image_path).split('.')[0]

            # load annotations

            if not os.path.exists(new_annotation_path):
                actual_labels.append(0)

            else:
                # check if there are any boxes in the annotation file

                with open(new_annotation_path) as f:

                    lines = f.readlines()

                    if len(lines) == 0:
                        actual_labels.append(0)

                    else:
                        actual_labels.append(1)

            # load predictions

            prediction_file_path = os.path.join(
                predictions_dir, base_name + '.txt')

            if not os.path.exists(prediction_file_path):
                predicted_labels.append(0)

            else:

                with open(prediction_file_path) as f:

                    lines = f.readlines()

                    if len(lines) == 0:
                        predicted_labels.append(0)

                    else:
                        predicted_labels.append(1)

    # calculate accuracy
    correct = 0
    total = len(actual_labels)

    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == predicted:
            correct += 1

    accuracy = correct / total

    return accuracy


# object count per image: actual vs predicted

def calculate_mae_pest_count_per_image(csv_mapping_file: str, predictions_dir: str) -> float:
    """
    Calculate mean absolute error for pest count per image

    Parameters:
        csv_mapping_file: path to csv mapping file (interal use only: not for participants)
        predictions_dir: path to directory containing prediction files (submitted by participants)

    Returns:
        mae: mean absolute error for pest count per image
    """

    # first list for actual count and second list for predicted count
    # here, count is the number of pests present in the image (any class)

    actual_counts = []
    predicted_counts = []

    with open(csv_mapping_file) as f:

        reader = csv.reader(f)
        next(reader)

        for row in reader:

            local_image_path = row[0]
            new_image_path = row[1]
            new_annotation_path = row[2]

            # base_name from new_image_path
            base_name = os.path.basename(new_image_path).split('.')[0]

            # load annotations

            if not os.path.exists(new_annotation_path):
                actual_counts.append(0)

            else:
                # check if there are any boxes in the annotation file

                with open(new_annotation_path) as f:

                    lines = f.readlines()

                    if len(lines) == 0:
                        actual_counts.append(0)

                    else:
                        actual_counts.append(len(lines))

            # load predictions

            prediction_file_path = os.path.join(
                predictions_dir, base_name + '.txt')

            if not os.path.exists(prediction_file_path):
                predicted_counts.append(0)

            else:

                with open(prediction_file_path) as f:

                    lines = f.readlines()

                    if len(lines) == 0:
                        predicted_counts.append(0)

                    else:
                        predicted_counts.append(len(lines))

    # calculate MAE

    total = len(actual_counts)
    mae = sum([abs(actual - predicted)
               for actual, predicted in zip(actual_counts, predicted_counts)]) / total

    return mae


# evualuate function considering all metrics

def evaluate(predictions_class_agnostic_dir: str,  csv_mapping_file: str, weights: List[float] = [0.50, 0.25, 0.25]) -> float:

    # calculate mAP
    map = calculate_mAP(csv_mapping_file, predictions_class_agnostic_dir)
    print(f"mAP: {map}")

    # calculate image level classification
    classification_acc = calculate_image_level_classification(
        csv_mapping_file, predictions_class_agnostic_dir)
    print(f"Image level classification accuracy: {classification_acc}")

    # calculate pest count per image
    pest_count_mae = calculate_mae_pest_count_per_image(
        csv_mapping_file, predictions_class_agnostic_dir)
    print(f"Pest count MAE: {pest_count_mae}")
    inverted_pest_count_mae = 1 / (pest_count_mae + 1)

    # calculate weighted average of all metrics
    weighted_avg_score = weights[0] * map + weights[1] * \
        classification_acc + weights[2] * inverted_pest_count_mae

    print(f"Weighted average score: {weighted_avg_score}")

    return weighted_avg_score
