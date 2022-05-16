from time import perf_counter as timer
from pathlib import Path
from typing import Any, Dict
import numpy as np 
import cv2
from yolov5facedetector.face_detector import YoloDetector
import face_detection
import logging
import warnings
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

class FailoverModel:
    """Face Detection using Failover Model"""
    def __init__(self, 
                 yolo_config:Dict[str, Any]=dict(yolo_type='yolov5l', confidence_threshold=0.6, target_size=720, gpu=0),
                 dsfd_config:Dict[str, Any]=dict(confidence_threshold=.65, nms_iou_threshold=.3)
                 )->None:
        """Initialize Failover Face Detection Model
        Input: Config of yolo & dsfd model"""
        warnings.filterwarnings('ignore')
        logger.info('Load YOLOv5-Face & DSFD Model to memory')
        start = timer()
        self.yoloface = YoloDetector(
            yolo_type=yolo_config['yolo_type'], 
            target_size=yolo_config['target_size'], 
            gpu=yolo_config['gpu'])
        self.dsfd = face_detection.build_detector(
            "DSFDDetector", 
            confidence_threshold=dsfd_config['confidence_threshold'], 
            nms_iou_threshold=dsfd_config['nms_iou_threshold'])
        logger.info(f'Load model done in {timer()-start}')
        self._yolo_config = yolo_config
        self._dsfd_config = dsfd_config

    def _correcting_rotation(self, face_cropped:np.array, face_keypoints:list)->np.array:
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

    def _filter_bbox(self, img:np.array, bboxes:list)->int:
        """Filter Bounding Box if more than one, use the one that closest to the center of images
        Return bbox index with most closest to the center"""    
        if len(bboxes)==1:
            return 0
        center_img = [int(img.shape[0]/2), int(img.shape[1]/2)]
        center_bbox = [[int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1])] for bbox in bboxes]
        l2_distances = [np.linalg.norm(np.array(bbox)-np.array(center_img)) for bbox in center_bbox]
        return np.argmin(l2_distances)

    def detect_face(self, rgb_img:np.array)->Dict[str, Any]:
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
                'rotation_degree: 0, 90, 180, or 270 degree,
                'bounding_box': [xmin,ymin,xmax,ymax] or [] empty list if not detected,
                'conf': confidence level or 0 if not detected, 
                'face_img': rgb_face_img or np.nan if not detected
            }
        """    
        # Try Yolo first
        for degree in [0,1,3,2]:
            img_rot = np.rot90(rgb_img, k=degree)
            yolo_bboxes, yolo_confs, yolo_points = self.yoloface.predict(img_rot, conf_thres = self._yolo_config['confidence_threshold'])
            yolo_bboxes, yolo_confs, yolo_points = yolo_bboxes[0], yolo_confs[0], yolo_points[0]
            if len(yolo_bboxes) != 0:
                bbox_idx = self._filter_bbox(img_rot, yolo_bboxes)
                yolo_bboxes = yolo_bboxes[bbox_idx]
                yolo_points = yolo_points[bbox_idx]
                yolo_confs = yolo_confs[bbox_idx]
                xmin,ymin,xmax,ymax = np.array(yolo_bboxes).astype(int)
                face_cropped = img_rot[ymin:ymax, xmin:xmax]
                # Check if face have correct output
                if (face_cropped.shape[0] <= 0) | (face_cropped.shape[1] <= 0):
                    continue
                face_cropped = self._correcting_rotation(face_cropped, yolo_points)
                return dict(
                    detector = 'yolo',
                    rotation_degree = degree*90,
                    bounding_box = [xmin,ymin,xmax,ymax],
                    conf = yolo_confs,
                    face_img = face_cropped
                )
        else:
            # Try DSFD        
            try:
                for degree in [0,1,3,2]:
                    img_rot = np.rot90(rgb_img, k=degree)
                    dsfd_res = self.dsfd.detect(img_rot)
                    if len(dsfd_res) != 0:                   
                        bbox_idx = _filter_bbox(img_rot, dsfd_res)
                        *dsfd_bbox, conf = dsfd_res[bbox_idx]
                        xmin,ymin,xmax,ymax = np.array(dsfd_bbox).astype(int)
                        face_cropped = rgb_img[ymin:ymax, xmin:xmax]
                        # Check if face have correct output
                        if (face_cropped.shape[0] <= 0) | (face_cropped.shape[1] <= 0):
                            continue
                        return dict(
                            detector = 'dsfd',
                            rotation_degree = degree*90,
                            bounding_box = [xmin,ymin,xmax,ymax],
                            conf = conf,
                            face_img = face_cropped
                        )                
                else:
                    return dict(
                        detector = 'none',
                        rotation_degree = 0,
                        bounding_box = [],
                        conf = 0,
                        face_img = np.nan
                    )
            except:
                return dict(
                        detector = 'none',
                        rotation_degree = 0,
                        bounding_box = [],
                        conf = 0,
                        face_img = np.nan
                    )
                    
# if __name__ == '__main__':
#     import cv2
#     img_rgb = cv2.imread('/ai-transfer-data-clone/face-search-fraud/export_search_group/apr22_kyc_facesearch_2022-05-13/group_1/id_search_ip-10-11-5-24.ap-southeast-1.compute.internal_V2_SELFIE_6282317517505_1649178431009.jpg')[...,::-1]            
#     model = FailoverModel()
#     res = model.detect_face(img_rgb)
#     print(res)