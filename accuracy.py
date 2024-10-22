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
import matplotlib.pyplot as plt
from sklearn.metrics import auc

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


precision = []
recall = []
errors = []
id_acc = []

def get_boxes(pic_path):
    pic = cv2.imread(pic_path)

    detections = detector(pic, verbose=False, iou=0.5, conf=0.03)[0].cpu().boxes

    xyxy = [
        [
            int(max(det[0], 0)),
            int(max(det[1], 0)),
            int(min(det[2], pic.shape[1] - 1)),
            int(min(det[3], pic.shape[0] - 1))
        ]
        for det in [
            [int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections
        ]
    ]

    return xyxy

def get_intersection_area(topleft1, bottomright1, topleft2, bottomright2):
    intersection_topleft = (max(topleft1[0], topleft2[0]), max(topleft1[1], topleft2[1]))
    intersection_bottomright = (min(bottomright1[0], bottomright2[0]), min(bottomright1[1], bottomright2[1]))

    intersection_width = intersection_bottomright[0] - intersection_topleft[0]
    intersection_height = intersection_bottomright[1] - intersection_topleft[1]

    if intersection_width <= 0 or intersection_height <= 0:
        return 0  # No intersection
    
    return intersection_width * intersection_height

def get_box_stat(xyxy, annotation_path):
    with open(annotation_path, 'r') as file:
        annotations = json.load(file)  
    actual_points = []
    actual_ids = []
    for marker in annotations["markers"]:
        corners = []
        for corner in marker["corners"]:
            corners.append([round(corner["x"], 2), round(corner["y"], 2)])
        actual_points.append(sorted(corners))
        actual_ids.append(marker["id"])
    # actual_points.sort()
    # zipped_lists = list(zip(actual_points, actual_ids))
    # zipped_lists.sort(key=lambda x:x[0])
    # actual_points, actual_ids = zip(*zipped_lists)
    # actual_points = list(actual_points)
    # actual_ids = list(actual_ids)

    tp = 0
    fn = 0
    N = len(actual_points) # detected markers

    match = {}
    
    for i, box in enumerate(actual_points):
        leftx = rightx = lefty = righty = None
        for point in box:
            if leftx is None or leftx > point[0]:
                leftx = point[0]
            if rightx is None or rightx < point[0]:
                rightx = point[0]
            if lefty is None or lefty > point[1]:
                lefty = point[1]
            if righty is None or righty < point[1]:
                righty = point[1]
        true_area = (rightx - leftx) * (righty - lefty)

        found = False

        for j, points in enumerate(xyxy):
            box_area = (points[2] - points[0]) * (points[3] - points[1])
            intersection_area = get_intersection_area((points[0], points[1]), (points[2], points[3]), (leftx, lefty), (rightx, righty))
            if intersection_area == 0:
                continue
            
            union_area = true_area + box_area - intersection_area

            iou = intersection_area / union_area if union_area > 0 else 0
            if iou >= 0.5:
                match[j] = i
                tp+=1
                found = True
                break
        if not found:
            fn += 1
    # fp = 0
    # for i in range(len(xyxy)):
    #     if i not in match:
    #         fp+=1
    
    # print("True positive is",tp/N)
    # print("False positive is",fp/N)
    # print("False negative is",fn/N)

    # print("Precision is", tp / len(xyxy))
    # print("Recall is", tp / (tp + fn))
    precision.append(tp/len(xyxy))
    recall.append(tp/(tp + fn))

    true_boxes = []
    true_annots = []
    true_ids = []
    for box_idx, annot_idx in match.items():
        true_boxes.append(xyxy[box_idx])
        true_annots.append(actual_points[annot_idx])
        true_ids.append(actual_ids[annot_idx])
    true_boxes.sort()
    zipped_lists = list(zip(true_annots, true_ids))
    zipped_lists.sort(key=lambda x:x[0])
    true_annots, true_ids = zip(*zipped_lists)
    true_annots = list(true_annots)
    true_ids = list(true_ids)
    return true_boxes, true_annots, true_ids


def get_pred_and_actual(pic_path, annotation_path):
    @tf.function(reduce_retracing=True)
    def refine_corners(crops):
        return regressor(crops)

    @tf.function(reduce_retracing=True)
    def decode_markers(markers):
        return decoder(markers)

    pic = cv2.imread(pic_path)

    xyxy = get_boxes(pic_path)
    true_xyxy, annot_xyxy, pred_ids = get_box_stat(xyxy, annotation_path)
    true_xyxy = [
        [
            int(max(x1 - (0.2 * (x2 - x1) + 0.5), 0)),
            int(max(y1 - (0.2 * (y2 - y1) + 0.5), 0)),
            int(min(x2 + (0.2 * (x2 - x1) + 0.5), pic.shape[1] - 1)),
            int(min(y2 + (0.2 * (y2 - y1) + 0.5), pic.shape[0] - 1))
        ]
        for (x1, y1, x2, y2) in true_xyxy  # Loop through each bounding box
    ]

    crops_ori = [
        cv2.resize(pic[det[1] : det[3], det[0] : det[2]], (64, 64)) for det in true_xyxy
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
    true_xyxy, crops_ori, corners = zip(
        *[
            (det, crop, cs)
            for det, crop, cs, k in zip(true_xyxy, crops_ori, corners, keep)
            if k == True
        ]
    )

    corners = [
        ordered_corners([c[0] for c in cs], [c[1] for c in cs]) for cs in corners
    ]

    pred_points = []
    for cs, det in zip(corners, true_xyxy):

        cs = [(cs[i], cs[i + 1]) for i in range(0, 8, 2)]

        width = det[2] - det[0]
        height = det[3] - det[1]
        points = []
        for i in range(0, 4):
            p1 = (int(det[0] + cs[i][0] * width), int(det[1] + cs[i][1] * height))
            points.append([p1[0], p1[1]])
        pred_points.append(sorted(points))
    pred_points.sort()

    markers = []

    for crop, cs in zip(crops_ori, corners):
        marker = marker_from_corners(crop, cs, 32)
        markers.append(norm(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))
    decoder_out = np.round(decode_markers(np.array(markers)).numpy())

    ids, dists = zip(*[find_id(out) for out in decoder_out])

    return pred_points, annot_xyxy, pred_ids, ids

def find_accuracy(pred_points, actual_points):
    dist = 0
    for i in range(len(pred_points)):
        for j in range(len(pred_points[i])):
            dist += abs(pred_points[i][j][0] - actual_points[i][j][0]) + abs(pred_points[i][j][1] - actual_points[i][j][1])
    return dist/(8 * len(pred_points))

def get_id_accuracy(pred_ids, actual_ids):
    correct = 0
    for i in range(len(pred_ids)):
        if pred_ids[i] == actual_ids[i]:
            correct +=1
    return correct/len(pred_ids)



all_images = os.listdir("shadow aruco")
images = ["shadow aruco/" + file for file in all_images if file.endswith(".png")]
images.sort()

all_annotations = os.listdir("shadow aruco/corrected_annotations")
annotations = ["shadow aruco/corrected_annotations/" + file for file in all_annotations if file.endswith(".json")]
annotations.sort()

# for i in range(len(all_images)):
#     true_positives, true_annotations, pred_ids, true_ids = get_pred_and_actual(images[i], annotations[i]) 
#     print("Mean absolute error for true positives is", find_accuracy(true_positives, true_annotations))
#     id_ac = get_id_accuracy(pred_ids, true_ids)
#     print("Id decoding accuracy is", id_ac)
#     print("---------------------------------------")

for i in range(len(images)):
    true_positives, true_annotations, pred_ids, true_ids = get_pred_and_actual(images[i], annotations[i]) 
    errors.append(find_accuracy(true_positives, true_annotations))
    # id_acc.append(get_id_accuracy(pred_ids, true_ids))

precision = np.array(precision)
recall = np.array(recall)
errors = np.array(errors)
id_acc = np.array(id_acc)

print("Precision is", precision.mean())
print("Recall is", recall.mean())
print("Error is", errors.mean())

# precision_value = 0.9858491241254674 # calculated
# recall_value = 0.9411491044428936
# error_value = 2.171253612286761
# print("Id accuracy is", id_acc.mean())
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

print(auc(recall, precision))


# Command to run is python3 accuracy.py
#compute precision and recall *