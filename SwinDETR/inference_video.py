""" Inference - input: video or real-time data """


import torch
import torchvision.transforms as T

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from time import time
from PIL import Image
import cv2
import csv
import threading

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models import SwinDETR
from args import base_parser
from utils.box_ops import rescale_bboxes


# real-world heights in meters for each class
REAL_HEIGHTS = {
    "person": 1.7,        # Average height of a person
    "bicycle": 1.5,       # Average height of a bicycle
    "car": 1.5,           # Average height of a car
    "motorcycle": 1.1,    # Average height of a motorcycle
    "bus": 3.0,           # Average height of a bus
    "truck": 3.0          # Average height of a truck    
}


# initialize CSV file (for recording data)
def initialize_csv(file_path):
    with open(file_path, mode='w', newline='') as csv_file:
        fieldnames = ['timestamp', 'label_id', 'confidence', 'km/h', 'xmin', 'ymin', 'xmax', 'ymax']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


# record detection result to CSV file
def log_detections(file_path, detections):
    """
    detections: list of tuples [(prob, label_id, km/h, distance), (xmin, ymin, xmax, ymax), ...]
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(file_path)

    try:
        with open(file_path, mode='a', newline='') as csv_file:
            fieldnames = ['timestamp', 'label_id', 'confidence', 'km/h', 'distance_m', 'xmin', 'ymin', 'xmax', 'ymax']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writeheader()

            for (prob, label_with_id, kmh, distance), (xmin, ymin, xmax, ymax) in detections:
                writer.writerow({
                    'timestamp': timestamp,
                    'label_id': label_with_id,
                    'confidence': f"{prob.max().item():.2f}",
                    'km/h': f"{kmh:.2f}",
                    'distance_m': f"{distance:.2f}",
                    'xmin': f"{xmin.item():.2f}",
                    'ymin': f"{ymin.item():.2f}",
                    'xmax': f"{xmax.item():.2f}",
                    'ymax': f"{ymax.item():.2f}"
                })
    except Exception as e:
        print(f"Error recording CSV: {e}")


# initialize model for inference
def inference_model(args):
    model = SwinDETR(args)
    
    # load pretrained weights
    assert args.weight and os.path.exists(args.weight), "Model weight file must be specified and exist"
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))

    model.eval()

    return model


# detect object from image
def detect(im, model, transform):
    img = transform(im).unsqueeze(0)
    outputs = model(img)

    probas = outputs['class'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8 # keep only predictions with 0.8+ confidence

    bboxes_scaled = rescale_bboxes(outputs['bbox'][0, keep], im.size)
    
    filtered_probas = []
    filtered_boxes = []
    label_counts = {}
    for i, p in enumerate(probas[keep]):
        cl = p.argmax()
        if cl < len(CLASSES) and CLASSES[cl] in REAL_HEIGHTS: # keep only the boxes and probabilities of the desired classes
            label = CLASSES[cl]
            # track the number of detected objects for each class
            if label not in label_counts:
                label_counts[label] = 1
            else:
                label_counts[label] += 1
            label_with_id = f"{label}_{label_counts[label]}" # add a unique Id for each object
            
            filtered_probas.append((p, label_with_id))
            filtered_boxes.append(bboxes_scaled[i])

    return filtered_probas, filtered_boxes


# real-time inference and calculate speed of each object
def real_time_inference(args):
    model = inference_model(args)
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # initialize background subtraction
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=250,       # number of frame for background learning
        varThreshold=16,   # threshold for background learning   
        detectShadows=True # detect shadow or not
    )

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(args.camera_id)


    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # initialize CSV file
    csv_file_path = './record/detections.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    initialize_csv(csv_file_path)

    # tracking data: {label_with_id: (last_position, bbox_last_height, last_time, last_km/h)}
    tracking_data = defaultdict(lambda: (None, None, None, 0.0)) # initial speed is zero

    pixels_per_meter = args.pixels_per_meter  # scale factor
    max_speed_change_kmh = args.max_speed_change_kmh  # maximum speed change limit value (km/h per frame)
    max_speed_kmh = args.max_speed_kmh  # maximum speed limit value
    focal_length = args.focal_length # focal distance (in pixels)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # create foreground mask
        fg_mask = back_sub.apply(frame)

        # convert the frame to PIL image for model inference
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        scores, boxes = detect(pil_image, model, transform)

        detections = []
        current_time = time()

        for (prob, label_with_id), (xmin, ymin, xmax, ymax) in zip(scores, boxes):
            """ bounding box area foreground pixel check """
            ixmin, iymin = int(xmin), int(ymin)
            ixmax, iymax = int(xmax), int(ymax)

            # clamp the area to avoid going out of the image range
            h, w = fg_mask.shape[:2]
            ixmin = max(0, min(ixmin, w-1))
            ixmax = max(0, min(ixmax, w-1))
            iymin = max(0, min(iymin, h-1))
            iymax = max(0, min(iymax, h-1))

            if ixmax <= ixmin or iymax <= iymin:
                continue  # skip invalid bbox

            roi_fg = fg_mask[iymin:iymax, ixmin:ixmax]  # bbox area foreground mask
            fg_count = cv2.countNonZero(roi_fg)
            
            bbox_area = (ixmax - ixmin) * (iymax - iymin)
            threshold_pixels = bbox_area * 0.1  # "stop" if less than 10% of the bbox area
            if fg_count < threshold_pixels:
                continue

            """ calculate distance of object """
            # extract class
            label = label_with_id.split('_')[0]
            real_height = REAL_HEIGHTS.get(label)
            bbox_height = (ymax - ymin).item()

            # calulate distance (pinhole camera equation)
            distance = (real_height * focal_length) / bbox_height # in meters

            """ calculate speed of object """
            # calculate center point
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            current_position = (x_center.item(), y_center.item())

            # calculate bounding box size (by height)
            bbox_height = (ymax - ymin).item()

            last_position, bbox_last_height, last_time, last_speed_kmh = tracking_data[label_with_id]

            if last_position is not None and bbox_last_height is not None and last_time is not None:
                # calculate position change (x, y)
                dx = current_position[0] - last_position[0]
                dy = current_position[1] - last_position[1]
                distance_pixels = (dx**2 + dy**2)**0.5
                distance_meters = distance_pixels * pixels_per_meter # scaling to meter
                time_elapsed = current_time - last_time
                # calculate speed for x and y (plane speed)
                speed_ms = distance_meters / time_elapsed if time_elapsed > 0 else 0.0
                speed_kmh = speed_ms * 3.6

                # bbox height size change (compared to the previous frame)
                size_change = bbox_height - bbox_last_height
                depth_change_meters = size_change * pixels_per_meter # scaling to meter
                # calculate speed for z (depth speed)
                speed_z_ms = depth_change_meters / time_elapsed if time_elapsed > 0 else 0.0
                speed_z_kmh = speed_z_ms * 3.6

                # calculate total speed (combine plane speed and depth speed)
                total_speed_kmh = (speed_kmh**2 + speed_z_kmh**2)**0.5

                # maximum speed change limit
                speed_change = total_speed_kmh - last_speed_kmh
                if abs(speed_change) > max_speed_change_kmh:
                    total_speed_kmh = last_speed_kmh + max_speed_change_kmh * (1 if speed_change > 0 else -1)

                # maximum speed limit value
                if total_speed_kmh > max_speed_kmh:
                    total_speed_kmh = max_speed_kmh
            else:
                total_speed_kmh = 0.0

            # update tracking data (update to current rate)
            tracking_data[label_with_id] = (current_position, bbox_height, current_time, total_speed_kmh)

            # add detection information with speed
            detections.append(((prob, label_with_id, total_speed_kmh, distance), (xmin, ymin, xmax, ymax)))

            # draw bounding boxes
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 2)

            # set text information (confidence, speed)
            confidence = prob.max().item()
            text = f'{label_with_id}: {confidence:.2f} prob, {total_speed_kmh:.2f} km/h, {distance:.2f} m'

            # calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            # text background box position
            text_x = int(xmin)
            text_y = int(ymin)
            box_coords = ((text_x, text_y), (text_x + text_width + 2, text_y + text_height + 2))
            # draw text background box
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 255, 255), cv2.FILLED)
            # draw text
            cv2.putText(frame, text, (text_x + 1, text_y + text_height + 1), font, font_scale, (0, 0, 0), thickness)

        if detections:
            threading.Thread(target=log_detections, args=(csv_file_path, detections)).start()

        cv2.imshow('Real-time Object Detection with Confidence and Speed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser('main.py', parents=[base_parser()])

    parser.add_argument('--swin_transformer', default='swin_t', type=str)
    parser.add_argument('--weight', default='./checkpoint/coco_swin_t.pt', type=str)
    parser.add_argument('--hidden_dim', default=768, type=int)

    parser.add_argument('--pixels_per_meter', default=0.01, type=float)
    parser.add_argument('--max_speed_change_kmh', default=10, type=int)
    parser.add_argument('--max_speed_kmh', default=60, type=int)

    # camera device ID (default: 0)
    parser.add_argument('--camera_id', default=0, type=int)
    # focal length of the camera in pixels
    parser.add_argument('--focal_length', default=800.0, type=float)

    # video file path
    parser.add_argument('--video_path', default='/Users/sekwang/Downloads/IMG_0526.MOV', type=str, help="Path to the input video file (e.g., .MOV, .MP4)")

    # COCO 2017 classes (part)
    CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], 
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    torch.set_grad_enabled(False)

    args = parser.parse_args()

    # check scale factor
    if args.pixels_per_meter <= 0:
        print("Error: pixels_per_meter must be greater than zero.")
        sys.exit(1)

    real_time_inference(args)