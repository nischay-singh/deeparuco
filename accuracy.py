from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf
from impl.aruco import find_id
from impl.heatmaps import pos_from_heatmap
from impl.losses import weighted_loss
from impl.utils import marker_from_corners, ordered_corners
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import json
import math
import os

norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
model_dir = "./models"

detector_model = f"{model_dir}/det_luma_bc_s.pt"
regressor_model = f"{model_dir}/reg_hmap_8.h5"
decoder_model = f"{model_dir}/dec_new.h5"

detector = YOLO(detector_model)
regressor = load_model(
    regressor_model,
    custom_objects={"weighted_loss": weighted_loss},
)
decoder = load_model(decoder_model)

def get_pred_and_actual(pic_path, annotation_path):
    @tf.function(reduce_retracing=True)
    def refine_corners(crops):
        return regressor(crops)

    @tf.function(reduce_retracing=True)
    def decode_markers(markers):
        return decoder(markers)

    pic = cv2.imread(pic_path)

    detections = detector(pic, verbose=False, iou=0.5, conf=0.03)[0].cpu().boxes

    xyxy = [
        [
            int(max(det[0] - (0.2 * (det[2] - det[0]) + 0.5), 0)),
            int(max(det[1] - (0.2 * (det[3] - det[1]) + 0.5), 0)),
            int(min(det[2] + (0.2 * (det[2] - det[0]) + 0.5), pic.shape[1] - 1)),
            int(min(det[3] + (0.2 * (det[3] - det[1]) + 0.5), pic.shape[0] - 1)),
        ]
        for det in [
            [int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections
        ]
    ]

    crops_ori = [
        cv2.resize(pic[det[1] : det[3], det[0] : det[2]], (64, 64)) for det in xyxy
    ]

    crops = [norm(crop) for crop in crops_ori]

    corners = refine_corners(np.array(crops)).numpy()

    area = 75
    kp_params = cv2.SimpleBlobDetector_Params()
    if area > 0:
        kp_params.filterByArea = True
        kp_params.minArea = area * 0.8
        kp_params.maxArea = area * 1.2
    kp_detector = cv2.SimpleBlobDetector_create(kp_params)

    corners = [
        [(x, y) for x, y in zip(*pos_from_heatmap(pred, kp_detector))]
        for pred in corners
    ]

    keep = [len(cs) == 4 for cs in corners]
    xyxy, crops_ori, corners = zip(
        *[
            (det, crop, cs)
            for det, crop, cs, k in zip(xyxy, crops_ori, corners, keep)
            if k == True
        ]
    )

    corners = [
        ordered_corners([c[0] for c in cs], [c[1] for c in cs]) for cs in corners
    ]
    pred_points = []
    for cs, det in zip(corners, xyxy):

        cs = [(cs[i], cs[i + 1]) for i in range(0, 8, 2)]

        width = det[2] - det[0]
        height = det[3] - det[1]
        points = []
        for i in range(0, 4):
            p1 = (int(det[0] + cs[i][0] * width), int(det[1] + cs[i][1] * height))
            points.append([p1[0], p1[1]])
        pred_points.append(sorted(points))
    pred_points.sort()

    with open(annotation_path, 'r') as file:
        annotations = json.load(file)  
    actual_points = []
    for marker in annotations["markers"]:
        corners = []
        for corner in marker["corners"]:
            corners.append([round(corner["x"], 2), round(corner["y"], 2)])
        actual_points.append(sorted(corners))
    actual_points.sort()

    return pred_points, actual_points

#assumes predicted points has all the markers
def find_accuracy(pred_points, actual_points):
    if len(pred_points) != len(actual_points):
        return -1
    dist = 0
    for i in range(len(pred_points)):
        for j in range(len(pred_points[i])):
            dist += (pred_points[i][j][0] - actual_points[i][j][0]) ** 2 + (pred_points[i][j][1] - actual_points[i][j][1]) ** 2
    return math.sqrt(dist)

all_images = os.listdir("shadow aruco")
images = ["shadow aruco/" + file for file in all_images if file.endswith(".png")]
images.sort()

all_annotations = os.listdir("shadow aruco/corrected_annotations")
annotations = ["shadow aruco/corrected_annotations/" + file for file in all_annotations if file.endswith(".json")]
annotations.sort()

for i in range(len(images)):
    predicted_points, actual_points = get_pred_and_actual(images[i], annotations[i])
    print(find_accuracy(predicted_points, actual_points))

# image, annot = get_pred_and_actual(images[0], annotations[0])
# print(image)
# # print(annot)

# add decoder - if decoded corner and id is correct



# Command to run is python3 accuracy.py