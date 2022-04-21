from time import perf_counter as timer
from pathlib import Path
from typing import Any, Dict
import numpy as np 
import cv2
import sys
import face_detection
import logging
import warnings
warnings.filterwarnings('ignore')
# Verification
YOLOFACE_PATH = Path.home()/'.yoloface'
if not YOLOFACE_PATH.exists():
    raise FileNotFoundError(f"Path to YoloFace Model doesn't exists please download from GCS then place to ~/.yoloface")
sys.path.append(str(YOLOFACE_PATH))
from face_detector import YoloDetector

# Load Model to memory
logging.info('Load YOLOv5-Face & DSFD Model to memory')
start = timer()
dsfd = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
yoloface = YoloDetector(weights_name='yolov5l_state_dict.pt',config_name='yolov5l.yaml',target_size=720)
logging.info(f'Load model done in {timer()-start}')

def correcting_rotation(face_cropped:np.array, face_keypoints):
    """Check if face rotated, rotate cropped image based on conditions in eyes and nose
    Face Keypoints need to have points for left_eye, right_eye and nose"""
    left_eye, right_eye, nose, *others = face_keypoints
    if not (left_eye[1] < nose[1]) & (right_eye[1] < nose[1]):
        # if not check if both eyes in the right of nose
        if (left_eye[0] > nose[0]) & (right_eye[0] > nose[0]):
            rot = 1
        elif (left_eye[0] < nose[0]) & (right_eye[0] < nose[0]):
            rot = 3
        else:
            rot = 2
        face_cropped = np.rot90(face_cropped, k=rot)
    return face_cropped

def face_detection_failover(rgb_img:np.array)->Dict[str, Any]:
    """Face Detection using failover systems
    Detect_Face -> Yolov5, if detected -> correcting orientation
    if not detected try -> DSFD model
    else, return np.nan

    Parameters
    ----------
    rgb_img : np.array
        Numpy Array of RGB Image

    Returns
    -------
    Dict[str, Any]
        {
            'detector': 'yolo','dsfd', or 'none',
            'face_img': rgb_face_img or np.nan if not detected
        }
    """    
    # Try Yolo first
    for degree in [0,1,3,2]:
        img_rot = np.rot90(rgb_img, k=degree)
        yolo_bboxes, yolo_points = yoloface.predict(img_rot,conf_thres = 0.65)
        if len(yolo_bboxes[0]) != 0:
            yolo_bboxes = yolo_bboxes[0][0]
            yolo_points = yolo_points[0][0]
            xmin,ymin,xmax,ymax = np.array(yolo_bboxes).astype(int)
            face_cropped = img_rot[ymin:ymax, xmin:xmax]
            # Check if face have correct output
            if (face_cropped.shape[0] <= 0) | (face_cropped.shape[1] <= 0):
                continue
            face_cropped = correcting_rotation(face_cropped, yolo_points)
            return dict(
                detector = 'yolo',
                face_img = face_cropped
            )
    else:
        # Try DSFD
        try:
            dsfd_res = dsfd.detect(rgb_img)
            if len(dsfd_res) != 0:
                *dsfd_bbox, conf = dsfd_res[0]
                xmin,ymin,xmax,ymax = np.array(dsfd_bbox).astype(int)
                face_cropped = rgb_img[ymin:ymax, xmin:xmax]
                # Check if face have correct output
                if (face_cropped.shape[0] <= 0) | (face_cropped.shape[1] <= 0):
                    return dict(
                        detector = 'none',
                        face_img = np.nan
                    )
                return dict(
                    detector = 'dsfd',
                    face_img = face_cropped
                )                
            else:
                return dict(
                    detector = 'none',
                    face_img = np.nan
                )
        except:
            return dict(
                    detector = 'none',
                    face_img = np.nan
                )
                
    