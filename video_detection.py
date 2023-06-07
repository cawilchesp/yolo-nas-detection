from super_gradients.training import models

import cv2
import torch
import numpy as np
import yaml
import time

from set_color import set_color

# For debugging
from icecream import ic


def time_synchronized():
    """
    PyTorch accurate time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def draw_boxes(image: np.array, boxes: list, class_ids: list) -> None:
    """
    Draw bounding boxes on frame
    """
    for (box, class_id) in zip(boxes, class_ids):
        if class_id in detection_config['DETECTION']['CLASS_FILTER']:
            # Draw box
            x1, y1, x2, y2 = [int(j) for j in box]
            color = set_color(class_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_label(image: np.array, boxes: list, confidences: list, class_ids: list, class_names: list) -> None:
    """
    Draw object label on frame
    """
    for (box, confidence, class_id) in zip(boxes, confidences, class_ids):
        if class_id in detection_config['DETECTION']['CLASS_FILTER']:
            # Draw box
            x1, y1, _, _ = [int(j) for j in box]
            color = set_color(class_id)
            label = f'{class_names[class_id]} {confidence:.2f}'

            # Draw label
            t_size = cv2.getTextSize(label, 0, 1/3, 1)[0]
            cv2.rectangle(image, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (x1, y1 - 2), 0, 1/3, [225, 255, 255], 1, cv2.LINE_AA)


def write_csv(csv_path: str, boxes: list, confidences: list, class_ids: list, class_names: list, frame_number: int) -> None:
    """
    Write object detection results in csv file
    """
    for (box, confidence, class_id) in zip(boxes, confidences, class_ids):
        if class_id in detection_config['DETECTION']['CLASS_FILTER']:
            x1, y1, x2, y2 = [int(j) for j in box]

            # Save results in CSV
            with open(f'{csv_path}.csv', 'a') as f:
                f.write(f'{frame_number},{class_id},{class_names[class_id]},{x1},{y1},{x2-x1},{y2-y1},{confidence:.5f}\n')


def main():
    # Initialize YOLO-NAS Model
    device = torch.device('cuda:0')
    model = models.get(
            model_name=detection_config['YOLO_NAS']['YOLO_NAS_WEIGHTS'],
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
    output_file_name = f"{input_config['FOLDER']}{input_config['FILE']}/output_{input_config['FILE']}_{detection_config['YOLO_NAS']['YOLO_NAS_WEIGHTS']}"
    
    video_writer_flag = False
    if detection_config['SAVE']['VIDEO']:
        video_writer_flag = True
        video_writer = cv2.VideoWriter(f'{output_file_name}.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    
    # Run YOLOv8 inference
    print('***             Video Processing Start             ***')
    frame_number = 0

    while True:
        success, image = cap.read()
        if not success: break

        # Process YOLO-NAS detections
        t1 = time_synchronized()
        detections = list(model.predict(image))[0]
        t2 = time_synchronized()
        
        class_names = detections.class_names
        boxes = detections.prediction.bboxes_xyxy.tolist()
        confidences = detections.prediction.confidence.tolist()
        class_ids = detections.prediction.labels.astype(int).tolist()

        # Visualization
        if detection_config['SHOW']['BOXES']: draw_boxes(image, boxes, class_ids)
        if detection_config['SHOW']['LABELS']: draw_label(image, boxes, confidences, class_ids, class_names)
        
        # Increase frame number
        print(f'Progress: {frame_number}/{frame_count}, Inference time: {t2-t1:.2f} s')
        frame_number += 1

        # Visualization
        cv2.imshow('source', image)

        # Save in CSV
        if detection_config['SAVE']['CSV']: write_csv(output_file_name, boxes, confidences, class_ids, class_names, frame_number)

        # Save video
        if video_writer_flag: video_writer.write(image)
        
        # Stop if Esc key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    if video_writer_flag:
        video_writer.release()

if __name__ == "__main__":
    # Initialize Configuration File
    with open('detection_config.yaml', 'r') as file:
        detection_config = yaml.safe_load(file)

    with torch.no_grad():
        main()
        