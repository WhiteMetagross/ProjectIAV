#This program implements a vehicle path tracker using YOLO11m and YOLO11m-seg for detection and segmentation, 
#and a custom BoTSORT implementation for trackings. It uses a stacked algorithm with morphological operations to ensure 
#tracking and segmentation of vehicles in video frames.

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import argparse
from pathlib import Path
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from botsort import BoTSORT

#Ensure fastreid is installed and accessible.
import fastreid
from fastreid.config import get_cfg


#This class encapsulates the vehicle path tracking functionality.
class SegmentedVehiclePathTracker:
    
    #Initialize the tracker with model paths, confidence threshold, and device.
    def __init__(self, yolo_model_path=r"C:\Users\Xeron\OneDrive\Documents\Programs\RoadVehiclesYOLODatasetProTraining\RoadVehiclesYOLODatasetPro_TrainingOutput\train\weights\RoadVehiclesYOLO11m.pt", 
                 segmentation_model_path=r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-seg.pt",
                 reid_model_path=r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
                 reid_config_path=r"C:\Users\Xeron\OneDrive\Documents\Programs\VehiclePathBoTSORTTracker\veriwild_r50_ibn_config.yml",
                 conf_threshold=0.35, device='cuda'):
        
        self.device = device
        self.conf_threshold = conf_threshold
        
        self.vehicle_classes = [0, 1, 2]
        
        print(f"[VehicleTracker] Initializing on {device}")
        
        self.yolo_model = YOLO(yolo_model_path)
        print(f"[VehicleTracker] YOLO detection model loaded: {yolo_model_path}")
        
        self.segmentation_model = YOLO(segmentation_model_path)
        print(f"[VehicleTracker] YOLO segmentation model loaded: {segmentation_model_path}")
        
        self.tracker = BoTSORT(
            reid_model_path=reid_model_path,
            reid_config_path=reid_config_path,
            device=device
        )
        
        self.frame_count = 0
        self.total_processing_time = 0
        self.track_trails = {}
        self.track_segments = {}
        self.previous_masks = {}

    #Validate and adjust bounding box coordinates to ensure they are within frame bounds.    
    def validate_bbox(self, x1, y1, x2, y2, frame_width, frame_height):
        x1 = max(0, min(int(x1), frame_width - 1))
        y1 = max(0, min(int(y1), frame_height - 1))
        x2 = max(x1 + 1, min(int(x2), frame_width))
        y2 = max(y1 + 1, min(int(y2), frame_height))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return x1, y1, x2, y2

    #Detect vehicles in the frame using YOLO model.   
    def detect_vehicles(self, frame):
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        h, w = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if conf >= self.conf_threshold and class_id in self.vehicle_classes:
                        bbox = self.validate_bbox(x1, y1, x2, y2, w, h)
                        if bbox is not None:
                            detections.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, class_id])
        
        return detections
    
    #Extract segmentation masks for detected vehicles using YOLO segmentation model.
    def extract_roi_segmentation(self, frame, detections):
        if not detections:
            return {}
        
        segmentation_data = {}
        h, w = frame.shape[:2]
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection[:4])
            
            bbox = self.validate_bbox(x1, y1, x2, y2, w, h)
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = bbox
            roi_width = x2 - x1
            roi_height = y2 - y1
            
            if roi_width <= 0 or roi_height <= 0:
                continue
            
            min_crop_size = 224
            padding_ratio = 0.4
            base_padding = max(50, int(padding_ratio * max(roi_width, roi_height)))
            
            current_width = roi_width + 2 * base_padding
            current_height = roi_height + 2 * base_padding
            
            if current_width < min_crop_size:
                extra_pad_x = (min_crop_size - current_width) // 2
                base_padding += extra_pad_x
            
            if current_height < min_crop_size:
                extra_pad_y = (min_crop_size - current_height) // 2
                base_padding += extra_pad_y
            
            x1_exp = max(0, x1 - base_padding)
            y1_exp = max(0, y1 - base_padding)
            x2_exp = min(w, x2 + base_padding)
            y2_exp = min(h, y2 + base_padding)
            
            exp_bbox = self.validate_bbox(x1_exp, y1_exp, x2_exp, y2_exp, w, h)
            if exp_bbox is None:
                continue
                
            x1_exp, y1_exp, x2_exp, y2_exp = exp_bbox
            exp_width = x2_exp - x1_exp
            exp_height = y2_exp - y1_exp
            
            if exp_width <= 0 or exp_height <= 0:
                continue
            
            roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            
            if roi.size > 0:
                try:
                    crop_area = exp_width * exp_height
                    
                    if crop_area < 10000:
                        seg_conf = 0.08
                        iou_thresh = 0.2
                    elif crop_area < 25000:
                        seg_conf = 0.12
                        iou_thresh = 0.25
                    elif crop_area < 50000:
                        seg_conf = 0.15
                        iou_thresh = 0.3
                    else:
                        seg_conf = 0.18
                        iou_thresh = 0.35
                    
                    if exp_width < 150 or exp_height < 150:
                        roi = cv2.convertScaleAbs(roi, alpha=1.1, beta=5)
                    
                    seg_results = self.segmentation_model(roi, conf=seg_conf, iou=iou_thresh, classes=[0, 1, 2, 3, 4, 5, 6, 7])
                    
                    #Process segmentation results to find the best mask overlapping with the original bounding box.
                    if seg_results and len(seg_results) > 0:
                        seg_result = seg_results[0]
                        if hasattr(seg_result, 'masks') and seg_result.masks is not None:
                            
                            masks = seg_result.masks.data.cpu().numpy()
                            boxes = seg_result.boxes.xyxy.cpu().numpy() if seg_result.boxes is not None else None
                            confidences = seg_result.boxes.conf.cpu().numpy() if seg_result.boxes is not None else None
                            class_ids = seg_result.boxes.cls.cpu().numpy().astype(int) if seg_result.boxes is not None else None
                            
                            best_mask = None
                            best_conf = 0
                            best_overlap = 0
                            
                            roi_orig_x1 = x1 - x1_exp
                            roi_orig_y1 = y1 - y1_exp
                            roi_orig_x2 = x2 - x1_exp
                            roi_orig_y2 = y2 - y1_exp
                            
                            if confidences is not None and class_ids is not None and boxes is not None:
                                for j, (mask, conf, cls_id, box) in enumerate(zip(masks, confidences, class_ids, boxes)):
                                    if cls_id in [0, 1, 2, 3, 4, 5, 6, 7]:
                                        seg_x1, seg_y1, seg_x2, seg_y2 = box
                                        
                                        overlap_x1 = max(roi_orig_x1, seg_x1)
                                        overlap_y1 = max(roi_orig_y1, seg_y1)
                                        overlap_x2 = min(roi_orig_x2, seg_x2)
                                        overlap_y2 = min(roi_orig_y2, seg_y2)
                                        
                                        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                            original_area = (roi_orig_x2 - roi_orig_x1) * (roi_orig_y2 - roi_orig_y1)
                                            overlap_ratio = overlap_area / original_area if original_area > 0 else 0
                                            
                                            combined_score = conf * 0.7 + overlap_ratio * 0.3
                                            
                                            if combined_score > best_conf and overlap_ratio > 0.3:
                                                best_conf = combined_score
                                                best_mask = mask
                                                best_overlap = overlap_ratio
                                        
                            elif len(masks) > 0:
                                best_mask = masks[0]
                                best_conf = 0.5
                            
                            if best_mask is not None and best_mask.size > 0:
                                mask_h, mask_w = best_mask.shape
                                
                                if mask_h != exp_height or mask_w != exp_width:
                                    if exp_width > 0 and exp_height > 0:
                                        mask_resized = cv2.resize(best_mask.astype(np.float32), (exp_width, exp_height))
                                    else:
                                        continue
                                else:
                                    mask_resized = best_mask.astype(np.float32)
                                
                                mask_resized = (mask_resized > 0.5).astype(np.uint8)
                                
                                roi_mask_in_original = np.zeros((roi_height, roi_width), dtype=np.uint8)
                                
                                ix1 = max(x1, x1_exp)
                                iy1 = max(y1, y1_exp)
                                ix2 = min(x2, x2_exp)
                                iy2 = min(y2, y2_exp)

                                if ix1 < ix2 and iy1 < iy2:
                                    src_ix1 = ix1 - x1_exp
                                    src_iy1 = iy1 - y1_exp
                                    src_ix2 = ix2 - x1_exp
                                    src_iy2 = iy2 - y1_exp

                                    dst_ix1 = ix1 - x1
                                    dst_iy1 = iy1 - y1
                                    dst_ix2 = ix2 - x1
                                    dst_iy2 = iy2 - y1

                                    if (src_iy2 > src_iy1 and src_ix2 > src_ix1 and 
                                        dst_iy2 > dst_iy1 and dst_ix2 > dst_ix1 and
                                        src_iy1 >= 0 and src_ix1 >= 0 and
                                        src_iy2 <= mask_resized.shape[0] and src_ix2 <= mask_resized.shape[1] and
                                        dst_iy1 >= 0 and dst_ix1 >= 0 and
                                        dst_iy2 <= roi_mask_in_original.shape[0] and dst_ix2 <= roi_mask_in_original.shape[1]):
                                        
                                        roi_mask_in_original[dst_iy1:dst_iy2, dst_ix1:dst_ix2] = mask_resized[src_iy1:src_iy2, src_ix1:src_ix2]

                                full_mask = np.zeros((h, w), dtype=np.uint8)
                                if y1 < h and x1 < w and y2 > 0 and x2 > 0:
                                    y1_clip = max(0, y1)
                                    x1_clip = max(0, x1)
                                    y2_clip = min(h, y2)
                                    x2_clip = min(w, x2)
                                    
                                    roi_y1_clip = y1_clip - y1
                                    roi_x1_clip = x1_clip - x1
                                    roi_y2_clip = roi_y1_clip + (y2_clip - y1_clip)
                                    roi_x2_clip = roi_x1_clip + (x2_clip - x1_clip)
                                    
                                    if (roi_y2_clip <= roi_mask_in_original.shape[0] and 
                                        roi_x2_clip <= roi_mask_in_original.shape[1]):
                                        full_mask[y1_clip:y2_clip, x1_clip:x2_clip] = roi_mask_in_original[roi_y1_clip:roi_y2_clip, roi_x1_clip:roi_x2_clip]
                                
                                segmentation_data[i] = {
                                    'mask': full_mask,
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': best_conf,
                                    'roi_mask': roi_mask_in_original,
                                    'overlap_ratio': best_overlap
                                }
                                
                except Exception as e:
                    print(f"[Warning] Segmentation failed for detection {i}: {e}")
                    continue
        
        return segmentation_data
    
    #Estimate a vehicle mask from bounding box using edge detection and morphological operations.
    def estimate_vehicle_mask_from_bbox(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        bbox = self.validate_bbox(x1, y1, x2, y2, w, h)
        if bbox is None:
            return np.zeros((max(1, y2-y1), max(1, x2-x1)), dtype=np.uint8)
        
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        if roi_h <= 0 or roi_w <= 0 or roi.size == 0:
            return np.zeros((max(1, y2-y1), max(1, x2-x1)), dtype=np.uint8)
        
        #Estimate mask using edge detection and morphological operations.
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 1.0)
            
            edges = cv2.Canny(blurred, 30, 100)
            
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
            
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask, [largest_contour], 255)
            else:
                center_x, center_y = roi_w // 2, roi_h // 2
                ellipse_w = max(10, int(roi_w * 0.7))
                ellipse_h = max(10, int(roi_h * 0.8))
                cv2.ellipse(mask, (center_x, center_y), (ellipse_w//2, ellipse_h//2), 0, 0, 360, 255, -1)
            
            if mask.size > 0 and np.any(mask):
                kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth)
            
            return mask
            
        except Exception as e:
            print(f"[Warning] Mask estimation failed: {e}")
            return np.zeros((max(1, y2-y1), max(1, x2-x1)), dtype=np.uint8)
    
    #Process a single frame: detect vehicles, extract segmentation, and update tracker.
    def process_frame(self, frame):
        start_time = time.time()
        
        detections = self.detect_vehicles(frame)
        
        segmentation_data = self.extract_roi_segmentation(frame, detections)
        
        tracks = self.tracker.update(detections, frame, segmentation_data)
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.frame_count += 1
        
        return tracks, detections, segmentation_data, processing_time
    
    #Assign a unique color to each track ID for visualization.
    def get_track_color(self, track_id):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
            (0, 255, 255), (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
            (128, 0, 255), (0, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
            (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192),
            (0, 192, 192), (128, 128, 0), (128, 0, 128), (0, 128, 128), (64, 64, 64)
        ]
        return colors[int(track_id) % len(colors)]
    
    #Update the trail of a track with the new center point.
    def update_trail(self, track_id, center):
        if track_id not in self.track_trails:
            self.track_trails[track_id] = []
        
        self.track_trails[track_id].append(center)
        
        if len(self.track_trails[track_id]) > 75:
            self.track_trails[track_id] = self.track_trails[track_id][-75:]

    #Update the segmentation masks history for a track.
    def update_track_segments(self, track_id, mask):
        if track_id not in self.track_segments:
            self.track_segments[track_id] = []
        
        self.track_segments[track_id].append(mask)
        
        if len(self.track_segments[track_id]) > 10:
            self.track_segments[track_id] = self.track_segments[track_id][-10:]
            
        self.previous_masks[track_id] = mask
    
    #Draw tracking results, bounding boxes, trails, and segmentation masks on the frame.
    def draw_results(self, frame, tracks, detections, segmentation_data):
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        valid_tracks = []
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2 = track[:4]
                bbox = self.validate_bbox(x1, y1, x2, y2, w, h)
                if bbox is not None:
                    track_modified = list(track)
                    track_modified[:4] = bbox
                    valid_tracks.append(track_modified)
                else:
                    track_id = int(track[4]) if len(track) > 4 else -1
                    print(f"[Warning] Invalid bounding box from tracker for track ID {track_id}: {track[:4]}. Skipping.")

        track_to_detection_map = {}
        for track in valid_tracks:
            track_id = int(track[4])
            track_bbox = track[:4]
            
            best_iou = 0
            best_detection_idx = -1
            
            for j, detection in enumerate(detections):
                det_bbox = detection[:4]
                
                x1_int = max(det_bbox[0], track_bbox[0])
                y1_int = max(det_bbox[1], track_bbox[1])
                x2_int = min(det_bbox[2], track_bbox[2])
                y2_int = min(det_bbox[3], track_bbox[3])
                
                if x1_int < x2_int and y1_int < y2_int:
                    intersection = (x2_int - x1_int) * (y2_int - y1_int)
                    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                    union = det_area + track_area - intersection
                    
                    if union > 0:
                        iou = intersection / union
                        if iou > best_iou:
                            best_iou = iou
                            best_detection_idx = j
            
            if best_iou > 0.3:
                track_to_detection_map[track_id] = best_detection_idx
        
        #Draw segmentation masks first for better visibility.
        for track in valid_tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            color = self.get_track_color(track_id)
            
            final_mask_roi = None
            
            det_idx = track_to_detection_map.get(track_id)
            if det_idx is not None and det_idx in segmentation_data:
                seg_data = segmentation_data[det_idx]
                mask = seg_data.get('mask')
                if mask is not None and mask.size > 0 and np.sum(mask) > 0:
                    roi_y1 = max(0, y1)
                    roi_x1 = max(0, x1)
                    roi_y2 = min(h, y2)
                    roi_x2 = min(w, x2)
                    
                    if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                        final_mask_roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
                        
                        if final_mask_roi.size > 0 and np.any(final_mask_roi):
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            final_mask_roi = cv2.morphologyEx(final_mask_roi, cv2.MORPH_CLOSE, kernel)
                            final_mask_roi = cv2.morphologyEx(final_mask_roi, cv2.MORPH_OPEN, kernel)

            if final_mask_roi is None or final_mask_roi.size == 0 or not np.any(final_mask_roi):
                final_mask_roi = self.estimate_vehicle_mask_from_bbox(frame, x1, y1, x2, y2)
            
            if (final_mask_roi is not None and final_mask_roi.size > 0 and 
                final_mask_roi.shape == (y2 - y1, x2 - x1) and np.any(final_mask_roi)):
                
                full_mask = np.zeros((h, w), dtype=np.uint8)
                
                roi_y1 = max(0, y1)
                roi_x1 = max(0, x1)
                roi_y2 = min(h, y2)
                roi_x2 = min(w, x2)
                
                mask_y1 = roi_y1 - y1
                mask_x1 = roi_x1 - x1
                mask_y2 = mask_y1 + (roi_y2 - roi_y1)
                mask_x2 = mask_x1 + (roi_x2 - roi_x1)
                
                if (mask_y2 <= final_mask_roi.shape[0] and mask_x2 <= final_mask_roi.shape[1] and
                    mask_y1 >= 0 and mask_x1 >= 0):
                    full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = final_mask_roi[mask_y1:mask_y2, mask_x1:mask_x2]

                    mask_indices = np.where(full_mask > 0)
                    if len(mask_indices[0]) > 0:
                        overlay = annotated_frame[mask_indices[0], mask_indices[1]].astype(float)
                        overlay = overlay * 0.7 + np.array(color) * 0.3
                        annotated_frame[mask_indices[0], mask_indices[1]] = overlay.astype(np.uint8)
        
        #Draw bounding boxes, trails, and labels.
        for track in valid_tracks:
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = track[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)

                color = self.get_track_color(track_id)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.update_trail(track_id, (center_x, center_y))
                
                trail_points = self.track_trails.get(track_id, [])
                if len(trail_points) > 1:
                    for i in range(1, len(trail_points)):
                        cv2.line(annotated_frame, trail_points[i-1], trail_points[i], color, 2)
                
                class_id = 0
                if len(track) >= 6:
                    class_id = int(track[5])
                
                class_name = self.yolo_model.names.get(class_id, 'Vehicle')
                label = f"{class_name} #{track_id}"
                
                text_scale = 0.3
                text_thickness = 1
                text_padding = 3
                text_color = (255, 255, 255)

                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
                
                label_box_x1 = x1
                label_box_y1 = y1 - text_height - baseline - (text_padding * 2)
                label_box_x2 = x1 + text_width + text_padding * 2
                label_box_y2 = y1
                
                if label_box_y1 < 0:
                    label_box_y1 = y2
                    label_box_y2 = y2 + text_height + baseline + (text_padding * 2)

                cv2.rectangle(annotated_frame, (label_box_x1, label_box_y1), (label_box_x2, label_box_y2), color, -1)
                
                cv2.putText(annotated_frame, label, (x1 + text_padding, label_box_y1 + text_height + text_padding), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
                
                cv2.circle(annotated_frame, (center_x, center_y), 4, (0, 0, 0), -1)
                cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
                cv2.circle(annotated_frame, (center_x, center_y), 1, (255, 255, 255), -1)

        return annotated_frame
    
    #Process the entire video: read frames, process each frame, and write output video.
    def track_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video file {output_path}")
            cap.release()
            return False
        
        print(f"Processing video: {input_path}")
        print(f"Output: {output_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_num = 0
        
        #Process each frame in the video.
        try:
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break
                
                if current_frame is None or current_frame.size == 0:
                    continue
                    
                frame_num += 1
                
                tracks, detections, segmentation_data, processing_time = self.process_frame(current_frame)
                
                annotated_frame = self.draw_results(current_frame, tracks, detections, segmentation_data)
                
                fps_current = 1.0 / processing_time if processing_time > 0 else 0
                
                reid_status = "ON" if len(detections) > 0 else "OFF"
                
                cv2.putText(annotated_frame, f'FPS: {fps_current:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Tracks: {len(tracks)} | Detections: {len(detections)}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Frame: {frame_num}/{total_frames} | ReID: {reid_status}', 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Segments: {len(segmentation_data)}', 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                out.write(annotated_frame)
                
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
        
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            cap.release()
            out.release()
            
        avg_fps = self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
        print(f"Processing complete!")
        print(f"Average processing FPS: {avg_fps:.2f}")
        
        return True


#Main function to parse arguments and run the tracker.
def main():
    #Parse command line arguments for input/output paths and model configurations.
    parser = argparse.ArgumentParser(description="Enhanced Vehicle Path Tracker with ROI Segmentation")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--yolo-model", default=r"C:\Users\Xeron\OneDrive\Documents\Programs\RoadVehiclesYOLODatasetProTraining\RoadVehiclesYOLODatasetPro_TrainingOutput\train\weights\RoadVehiclesYOLO11m.pt", help="YOLO detection model path")
    parser.add_argument("--segmentation-model", default=r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-seg.pt", help="YOLO segmentation model path")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("Enhanced Vehicle Path Tracker with ROI Segmentation")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"YOLO Detection Model: {args.yolo_model}")
    print(f"YOLO Segmentation Model: {args.segmentation_model}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    try:
        tracker = SegmentedVehiclePathTracker(
            yolo_model_path=args.yolo_model,
            segmentation_model_path=args.segmentation_model,
            conf_threshold=args.conf_threshold,
            device=args.device
        )
        
        success = tracker.track_video(args.input, args.output)
        if success:
            print(f"Video processing completed successfully!")
            print(f"Output saved to: {args.output}")
        else:
            print("Video processing failed!")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTracking interrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())