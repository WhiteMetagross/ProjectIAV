#This program implements the BoTSORT multi-object tracker with ReID and mask integration.
#It uses Kalman Filters for motion prediction and FastReID for appearance feature extraction.

import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

#The newFastreID library is required for this implementation.
import fastreid
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer


#Define the KalmanBoxTracker class to manage individual object tracks.
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, feature=None, mask=None):
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        self.features = []
        if feature is not None:
            self.features.append(feature)
            
        self.masks = []
        if mask is not None:
            self.masks.append(mask)

    #Update the state vector with observed bbox.
    def update(self, bbox, feature=None, mask=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 50:
                self.features = self.features[-50:]
                
        if mask is not None:
            self.masks.append(mask)
            if len(self.masks) > 10:
                self.masks = self.masks[-10:]

    #Predict the next state.
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    #Return the current bounding box estimate.
    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)
    
    #Return the most recent feature.
    def get_feature(self):
        if self.features:
            return self.features[-1]
        return None
    
    #Return the most recent mask.
    def get_mask(self):
        if self.masks:
            return self.masks[-1]
        return None

    #Convert bbox from [x1,y1,x2,y2] to [x,y,s,r] format.
    @staticmethod
    def _convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    #Convert state vector to bbox [x1,y1,x2,y2] format.
    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

#Dqefine the FastReIDFeatureExtractor class to handle feature extraction using newFastReID.
class FastReIDFeatureExtractor:
    def __init__(self, model_path, config_path, device='cuda'):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.freeze()
        
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        checkpointer = Checkpointer(self.model)
        checkpointer.load(model_path)
        
        print(f"[FastReID] Loaded VeRiWild model on {self.device}")
    
    #Extract features from a list of image crops.
    def extract_features(self, crops):
        if not crops:
            return np.empty((0, 2048), dtype=np.float32)
        
        processed_crops = []
        for crop in crops:
            import cv2
            if crop.size == 0:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            
            crop_resized = cv2.resize(crop, (128, 256))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            crop_tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float() / 255.0
            processed_crops.append(crop_tensor)
        
        batch = torch.stack(processed_crops).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()
    
    #Extract features from crops with corresponding masks applied.
    def extract_masked_features(self, crops, masks):
        if not crops or not masks:
            return np.empty((0, 2048), dtype=np.float32)
        
        processed_crops = []
        import cv2
        
        for i, (crop, mask) in enumerate(zip(crops, masks)):
            if crop.size == 0:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
                
            if mask is not None and mask.size > 0:
                if len(mask.shape) == 2:
                    h, w = mask.shape
                    crop_h, crop_w = crop.shape[:2]
                    
                    if mask.shape != crop.shape[:2]:
                        mask = cv2.resize(mask.astype(np.float32), (crop_w, crop_h))
                    
                    mask_norm = mask.astype(np.float32)
                    if mask_norm.max() > 1.0:
                        mask_norm = mask_norm / 255.0
                    
                    mask_3ch = np.stack([mask_norm, mask_norm, mask_norm], axis=-1)
                    masked_crop = (crop.astype(np.float32) * mask_3ch).astype(np.uint8)
                else:
                    masked_crop = crop
            else:
                masked_crop = crop
                
            crop_resized = cv2.resize(masked_crop, (128, 256))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            crop_tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float() / 255.0
            processed_crops.append(crop_tensor)
        
        batch = torch.stack(processed_crops).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()

#Compute IoU between two sets of boxes.
def iou_batch(bb_test, bb_gt):
    
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

#Compute IoU between two sets of masks.
def mask_iou_batch(masks_test, masks_gt):
    if len(masks_test) == 0 or len(masks_gt) == 0:
        return np.zeros((len(masks_test), len(masks_gt)))
    
    iou_matrix = np.zeros((len(masks_test), len(masks_gt)))
    
    for i, mask_test in enumerate(masks_test):
        for j, mask_gt in enumerate(masks_gt):
            if mask_test is not None and mask_gt is not None:
                if mask_test.size > 0 and mask_gt.size > 0:
                    try:
                        if mask_test.shape != mask_gt.shape:
                            import cv2
                            mask_gt = cv2.resize(mask_gt.astype(np.float32), 
                                               (mask_test.shape[1], mask_test.shape[0]))
                        
                        mask_test_bin = (mask_test > 0.5).astype(bool)
                        mask_gt_bin = (mask_gt > 0.5).astype(bool)
                        
                        intersection = np.logical_and(mask_test_bin, mask_gt_bin).sum()
                        union = np.logical_or(mask_test_bin, mask_gt_bin).sum()
                        
                        if union > 0:
                            iou_matrix[i, j] = intersection / union
                    except Exception:
                        # Silently handle mask IoU computation errors
                        iou_matrix[i, j] = 0.0
    
    return iou_matrix

#Associate detections to trackers using a cost matrix based on IoU, ReID similarity, and mask IoU.
def associate_detections_to_trackers(detections, trackers, reid_features=None, track_features=None, 
                                   detection_masks=None, track_masks=None,
                                   iou_threshold=0.3, reid_threshold=0.7, mask_weight=0.2):
    
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    
    cost_matrix = 1 - iou_matrix
    association_method = "IoU only"
    
    if reid_features is not None and track_features is not None and len(reid_features) > 0 and len(track_features) > 0:
        try:
            reid_sim = np.dot(reid_features, track_features.T)
            
            if detection_masks is not None and track_masks is not None:
                mask_iou = mask_iou_batch(detection_masks, track_masks)
                if mask_iou.shape == iou_matrix.shape:
                    cost_matrix = 1 - (0.5 * iou_matrix + 0.3 * reid_sim + mask_weight * mask_iou)
                    association_method = "IoU + ReID + Mask"
                else:
                    cost_matrix = 1 - (0.7 * iou_matrix + 0.3 * reid_sim)
                    association_method = "IoU + ReID"
            else:
                cost_matrix = 1 - (0.7 * iou_matrix + 0.3 * reid_sim)
                association_method = "IoU + ReID"
        except Exception as e:
            cost_matrix = 1 - iou_matrix
            association_method = "IoU only"
    elif detection_masks is not None and track_masks is not None:
        try:
            mask_iou = mask_iou_batch(detection_masks, track_masks)
            if mask_iou.shape == iou_matrix.shape:
                cost_matrix = 1 - (0.8 * iou_matrix + mask_weight * mask_iou)
                association_method = "IoU + Mask"
        except Exception as e:
            pass
    
    if min(cost_matrix.shape) > 0:
        a = (cost_matrix > 1 - iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] > 1 - iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

#Define the main BoTSORT tracker class.
class BoTSORT:
    def __init__(self, reid_model_path, reid_config_path, max_disappeared=30, min_hits=3, 
                 iou_threshold=0.3, device='cuda'):
        
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.reid_extractor = FastReIDFeatureExtractor(
            reid_model_path, reid_config_path, device
        )
        
        self.trackers = []
        self.frame_count = 0
    
    #Update the tracker with new detections, frame, and optional segmentation data.
    def update(self, detections, frame=None, segmentation_data=None):
        self.frame_count += 1
        
        reid_features = None
        detection_masks = None
        
        if frame is not None and detections:
            crops = self._extract_crops(frame, detections)
            
            if segmentation_data and len(segmentation_data) > 0:
                detection_masks = []
                masked_crops = []
                
                for i, det in enumerate(detections):
                    if i in segmentation_data:
                        seg_data = segmentation_data[i]
                        roi_mask = seg_data.get('roi_mask')
                        
                        if roi_mask is not None and roi_mask.size > 0:
                            mask_norm = roi_mask.astype(np.float32)
                            if mask_norm.max() > 1.0:
                                mask_norm = mask_norm / 255.0
                            detection_masks.append(mask_norm)
                        else:
                            detection_masks.append(None)
                        
                        masked_crops.append(crops[i])
                    else:
                        detection_masks.append(None)
                        masked_crops.append(crops[i])
                
                reid_features = self.reid_extractor.extract_masked_features(masked_crops, detection_masks)
            else:
                reid_features = self.reid_extractor.extract_features(crops)
        
        trks = np.zeros((len(self.trackers), 5))
        track_features = []
        track_masks = []
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                track_features.append(self.trackers[t].get_feature())
                track_masks.append(self.trackers[t].get_mask())
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            if t < len(track_features):
                track_features.pop(t)
            if t < len(track_masks):
                track_masks.pop(t)
        
        #Convert features and masks to numpy arrays if valid.
        if track_features and all(f is not None for f in track_features):
            track_features = np.array(track_features)
        else:
            track_features = None
            
        if track_masks and not all(m is None for m in track_masks):
            pass
        else:
            track_masks = None
        
        if detections:
            dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])
        else:
            dets = np.empty((0, 5))
            
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, reid_features, track_features, detection_masks, track_masks, self.iou_threshold
        )
        
        for m in matched:
            feature = reid_features[m[0]] if reid_features is not None else None
            mask = detection_masks[m[0]] if detection_masks is not None else None
            self.trackers[m[1]].update(dets[m[0], :4], feature, mask)
        
        for i in unmatched_dets:
            feature = reid_features[i] if reid_features is not None else None
            mask = detection_masks[i] if detection_masks is not None else None
            trk = KalmanBoxTracker(dets[i, :4], feature, mask)
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_disappeared:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    #Extract image crops for each detection from the frame.
    def _extract_crops(self, frame, detections):
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
            else:
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
        return crops