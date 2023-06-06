from super_gradients.training import models

import cv2
import torch
import numpy as np
import yaml
from collections import deque

# For debugging
from icecream import ic


def draw_boxes(image: np.array, ds_output: np.array) -> None:
    """
    Draw bounding boxes on frame
    """
    for box in enumerate(ds_output):
        box_xyxy = box[1][0:4]
        class_id = box[1][-1]
        if class_id in detection_config['DETECTION']['CLASS_FILTER']:
            # Draw box
            x1, y1, x2, y2 = [int(j) for j in box_xyxy]
            color = set_color(class_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_label(image: np.array, ds_output: np.array) -> None:
    """
    Draw object label on frame
    """
    for box in enumerate(ds_output):
        box_xyxy = box[1][0:4]
        object_id = box[1][-2]
        class_id = box[1][-1]
        if class_id in detection_config['DETECTION']['CLASS_FILTER']:
            # Draw box
            x1, y1, _, _ = [int(j) for j in box_xyxy]
            color = set_color(class_id)
            label = f'{class_names[class_id]} {int(object_id)}'

            # Draw label
            t_size = cv2.getTextSize(label, 0, 1/3, 1)[0]
            cv2.rectangle(image, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (x1, y1 - 2), 0, 1/3, [225, 255, 255], 1, cv2.LINE_AA)



def main():
    # Initialize YOLO-NAS Model
    device = torch.device('cuda:0')
    model = models.get(
            model_name='yolo_nas_m',
            pretrained_weights='coco'
        ).to(device)
    
    # Initialize Input
    input_config = detection_config['INPUT']

    cap = cv2.VideoCapture(f"{input_config['FOLDER']}{input_config['FILE']}.avi")
    if not cap.isOpened():
        raise RuntimeError('Cannot open source')
    

    print('***                  Video Opened                  ***')


    fourcc = 'mp4v'  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output
    output_file_name = f"{input_config['FOLDER']}{input_config['FILE']}/output_{input_config['FILE']}_yolov8m"
    vid_writer = cv2.VideoWriter(f'{output_file_name}.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    
    # Run YOLOv8 inference
    print('***             Video Processing Start             ***')
    frame_number = 0

    while True:
        # print(f'Progress: {frame_number}/{frame_count}')

        success, image = cap.read()
        if not success: break

        # Process YOLO-NAS detections
        detections = list(model.predict(image))[0]

        boxes = detections.prediction.bboxes_xyxy.tolist()
        confidences = detections.prediction.confidence.tolist()
        class_ids = detections.prediction.labels.tolist()

        ds_output = list(zip(boxes, confidences, class_ids))

        

        # Visualization
        if detection_config['SHOW']['BOXES']: draw_boxes(image, ds_output)

        # Increase frame number
        frame_number += 1

        # Visualization
        cv2.imshow('source', image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break
    

if __name__ == "__main__":
    # Initialize Configuration File
    with open('detection_config.yaml', 'r') as file:
        detection_config = yaml.safe_load(file)

    # object tracks
    track_deque = {}

    with torch.no_grad():
        main()
        