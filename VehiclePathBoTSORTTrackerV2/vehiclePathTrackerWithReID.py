#This program uses a specialised YOLO model trained on the RoadVehiclesYOLODatasetPro dataset to detect vehicles in video frames.
#It then tracks these vehicles using the BoTSORT algorithm combined with newFastReID for reidentification.
#The program can process video files or webcam input, drawing bounding boxes and trails for each tracked vehicle.

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

import fastreid
from fastreid.config import get_cfg


#This class encapsulates the vehicle detection and tracking functionality.
class VehiclePathTracker:
    
    #Initialization of the tracker with model paths and parameters.
    def __init__(self, yolo_model_path=r"C:\Users\Xeron\OneDrive\Documents\Programs\RoadVehiclesYOLODatasetProTraining\RoadVehiclesYOLODatasetPro_TrainingOutput\train\weights\RoadVehiclesYOLO11m.pt", 
                 reid_model_path=r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
                 reid_config_path=r"C:\Users\Xeron\OneDrive\Documents\Programs\VehiclePathBoTSORTTracker\veriwild_r50_ibn_config.yml",
                 conf_threshold=0.35, device='cuda'):
        
        self.device = device
        self.conf_threshold = conf_threshold
        
        self.vehicle_classes = [0, 1, 2]
        
        print(f"[VehicleTracker] Initializing on {device}")
        
        self.yolo_model = YOLO(yolo_model_path)
        print(f"[VehicleTracker] YOLO model loaded: {yolo_model_path}")
        
        self.tracker = BoTSORT(
            reid_model_path=reid_model_path,
            reid_config_path=reid_config_path,
            device=device
        )
        
        self.frame_count = 0
        self.total_processing_time = 0
        self.track_trails = {}

    #Detect vehicles in a frame using the fine tuned YOLO model.    
    def detect_vehicles(self, frame):
        
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if conf >= self.conf_threshold and class_id in self.vehicle_classes:
                        detections.append([x1, y1, x2, y2, conf, class_id])
        
        return detections
    
    #Process a single frame: detect vehicles, update tracker, and measure processing time.
    def process_frame(self, frame):
        
        start_time = time.time()
        
        detections = self.detect_vehicles(frame)
        
        tracks = self.tracker.update(detections, frame)
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.frame_count += 1
        
        return tracks, detections, processing_time
    
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
    
    #Update the trail of a tracked vehicle with its new position.
    def update_trail(self, track_id, center):
        
        if track_id not in self.track_trails:
            self.track_trails[track_id] = []
        
        self.track_trails[track_id].append(center)
        
        if len(self.track_trails[track_id]) > 125:
            self.track_trails[track_id] = self.track_trails[track_id][-125:]
    
    #Draw bounding boxes and trails on the frame for each tracked vehicle.
    def draw_results(self, frame, tracks, detections):
        
        annotated_frame = frame.copy()
        
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = track[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)
                
                color = self.get_track_color(track_id)
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.update_trail(track_id, (center_x, center_y))
                
                trail_points = self.track_trails[track_id]
                if len(trail_points) > 1:
                    for i in range(1, len(trail_points)):
                        alpha = i / len(trail_points)
                        thickness = max(3, int(3 * alpha))
                        cv2.line(annotated_frame, trail_points[i-1], trail_points[i], color, thickness)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        
        return annotated_frame
    
    #Track vehicles in a video file, saving the annotated output.
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
        
        try:
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break
                
                if current_frame is None:
                    continue
                    
                frame_num += 1
                
                tracks, detections, processing_time = self.process_frame(current_frame)
                
                annotated_frame = self.draw_results(current_frame, tracks, detections)
                
                fps_current = 1.0 / processing_time if processing_time > 0 else 0
                
                reid_status = "ON" if len(detections) > 0 else "OFF"
                
                cv2.putText(annotated_frame, f'FPS: {fps_current:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Tracks: {len(tracks)} | Detections: {len(detections)}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Frame: {frame_num}/{total_frames} | ReID: {reid_status}', 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                out.write(annotated_frame)
                
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
        
        except Exception as e:
            print(f"Error during processing: {e}")
            return False
        
        finally:
            cap.release()
            out.release()
            
        avg_fps = self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
        print(f"Processing complete!")
        print(f"Average processing FPS: {avg_fps:.2f}")
        
        return True
    
    #Track vehicles using webcam input, displaying the annotated output in real time.
    def track_webcam(self):
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam tracking. Press 'q' to quit.")
        
        try:
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break
                
                tracks, detections, processing_time = self.process_frame(current_frame)
                
                annotated_frame = self.draw_results(current_frame, tracks, detections)
                
                fps = 1.0 / processing_time if processing_time > 0 else 0
                cv2.putText(annotated_frame, f'FPS: {fps:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Tracks: {len(tracks)} | Detections: {len(detections)}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow('Vehicle Tracking - Webcam', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


#The main function to parse arguments and run the tracker.
def main():
    
    #Argument parsing for input/output paths and model parameters.
    parser = argparse.ArgumentParser(description="Vehicle Path Tracker v3 with BoTSORT + FastReID")
    parser.add_argument("--source", choices=["video", "webcam"], default="video", 
                       help="Input source")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--yolo-model", default=r"C:\Users\Xeron\OneDrive\Documents\Programs\RoadVehiclesYOLODatasetProTraining\RoadVehiclesYOLODatasetPro_TrainingOutput\train\weights\RoadVehiclesYOLO11m.pt", help="YOLO model path")
    parser.add_argument("--conf-threshold", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("Vehicle Path Tracker v3")
    print("=" * 50)
    print(f"Source: {args.source}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    #Initialize and run the tracker based on the input source.
    try:
        tracker = VehiclePathTracker(
            yolo_model_path=args.yolo_model,
            conf_threshold=args.conf_threshold,
            device=args.device
        )
        
        if args.source == "video":
            success = tracker.track_video(args.input, args.output)
            if success:
                print(f"Video processing completed successfully!")
                print(f"Output saved to: {args.output}")
            else:
                print("Video processing failed!")
                return 1
        elif args.source == "webcam":
            tracker.track_webcam()
        
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