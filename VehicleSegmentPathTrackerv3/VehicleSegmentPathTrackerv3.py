#This program implements a vehicle path tracker using a custom YOLO11m fine tuned on the IDDD-Detection dataset 
#for detection and SAM 2.1 Base for segmentation, and a custom BoTSORT implementation for tracking. 
#It uses SAM 2.1 with bounding box prompts to ensure accurate tracking and segmentation of vehicles 
#in video frames with proper GPU memory management.

import cv2
import numpy as np
import torch
from ultralytics import YOLO, SAM
import time
import argparse
from pathlib import Path
import sys
import os
import warnings
import gc
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import List, Tuple, Dict, Optional
import colorsys

#Ensure the fastreid package is available for BoTSORT.
import fastreid
from fastreid.config import get_cfg
from botsort import BoTSORT

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#This function sets up PyTorch native optimization with CUDA streams and GPU optimizations.
def setup_pytorch_optimizations():
    if not torch.cuda.is_available():
        print("[Performance] CUDA not available: using CPU mode")
        return False, "cpu"
    
    #Suppress torch._dynamo errors for compatibility with SAM models.
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        print("[Performance] Torch dynamo errors suppressed for compatibility")
    except ImportError:
        pass
    
    #Check the GPU properties and compute capability.
    try:
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = device_props.major + device_props.minor / 10
        print(f"[Performance] GPU: {device_props.name}")
        print(f"[Performance] Compute Capability: {compute_capability}")
        print(f"[Performance] GPU Memory: {device_props.total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"[Performance] Error getting GPU properties: {e}")
        return False, "cpu"
    
    #Enable PyTorch native CUDA optimizations.
    try:
        #Enable PyTorch's native Scaled Dot Product Attention optimizations.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
        #Enhanced CUDA optimizations for maximum performance.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        #Enable JIT fusion for faster operations.
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        
        #Enable additional optimizations if available.
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(True)
        
        #Initialize CUDA context and warm up GPU.
        dummy_tensor = torch.randn(1, 1, device='cuda')
        torch.cuda.synchronize()
        del dummy_tensor
        
        #Check enabled backends.
        flash_sdp_enabled = torch.backends.cuda.flash_sdp_enabled()
        mem_efficient_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        math_enabled = torch.backends.cuda.math_sdp_enabled()
        
        print(f"[Performance] PyTorch Flash SDP: {flash_sdp_enabled}")
        print(f"[Performance] PyTorch Memory Efficient SDP: {mem_efficient_enabled}")
        print(f"[Performance] PyTorch Math SDP: {math_enabled}")
        print(f"[Performance] CUDA Memory Fraction: 95%")
        print(f"[Performance] JIT Fusion: Enabled")
        
        if flash_sdp_enabled:
            print("[Performance] PyTorch Flash SDP enabled: Maximum performance.")
            return True, "flash_sdp"
        elif mem_efficient_enabled:
            print("[Performance] Memory Efficient Attention enabled: Excellent performance.")
            return True, "mem_efficient"
        else:
            print("[Performance] Math SDP enabled: Standard performance")
            return True, "math_sdp"
            
    except Exception as e:
        print(f"[Performance] Error setting up CUDA optimizations: {e}")
        return False, "cpu"


#Initialize optimized PyTorch backends for maximum performance.
cuda_available, attention_backend = setup_pytorch_optimizations()


#This class encapsulates the vehicle path tracking functionality.
class SegmentedVehiclePathTracker:
    
    #Initialize the tracker with model paths, confidence threshold, and device.
    def __init__(self, yolo_model_path=r"C:\Users\Xeron\OneDrive\Documents\Programs\RoadVehiclesYOLODatasetProTraining\RoadVehiclesYOLODatasetPro_TrainingOutput\train\weights\RoadVehiclesYOLO11m.pt", 
                 sam_model_path="sam2.1_b.pt",
                 reid_model_path=r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
                 reid_config_path=r"C:\Users\Xeron\OneDrive\Documents\Programs\VehiclePathBoTSORTTracker\veriwild_r50_ibn_config.yml",
                 conf_threshold=0.35, device='cuda'):
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.attention_backend = attention_backend
        
        self.vehicle_classes = [0, 1, 2]
        
        print(f"[VehicleTracker] Initializing on {device}")
        
        self.yolo_model = YOLO(yolo_model_path)
        print(f"[VehicleTracker] YOLO detection model loaded: {yolo_model_path}")
        
        print(f"[VehicleTracker] Loading SAM 2.1 Base model with {attention_backend} acceleration...")
        
        self.sam_model = SAM(sam_model_path)
        self.sam_model.to(device)
        
        self.sam_model.eval()
        
        #Note: torch.compile causes issues with SAM's internal structures.
        print("[VehicleTracker] Using standard SAM optimization for maximum compatibility.")
        
        #Enable mixed precision for optimized backends.
        self.use_mixed_precision = attention_backend in ["flash_sdp", "mem_efficient."]
        if self.use_mixed_precision:
            print("[VehicleTracker] Mixed precision enabled for faster inference.")
        
        print(f"[VehicleTracker] SAM 2.1 Base loaded with {attention_backend} acceleration.")
        
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
        
        self.max_memory_usage = 0.85 if attention_backend in ["flash_sdp", "mem_efficient"] else 0.8
        self.cleanup_interval = 150 if attention_backend in ["flash_sdp", "mem_efficient"] else 100
        
        self.cuda_stream_inference = torch.cuda.Stream()
        self.cuda_stream_postprocess = torch.cuda.Stream()
        
        self.tensor_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        self.max_batch_size = self._calculate_optimal_batch_size()
        self._preallocate_tensors()
        
        self.cpu_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu_worker")
        
        self.preprocessing_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="preprocess")  # Increased workers
        self.frame_queue = queue.Queue(maxsize=5)
        self.preprocessing_active = True
        
        self.frame_skip_factor = 1  #Process every frame by default.
        self.detection_interval = 2  #Run detection every 2 frames, use tracking for others.
        self.segmentation_interval = 3  #Run segmentation every 3 frames for stable tracks.
        
        print(f"[Performance] Optimal batch size: {self.max_batch_size}")
        print(f"[Performance] CPU thread pool initialized with 3 workers")
        print(f"[Performance] CUDA streams initialized for parallel GPU operations")
        print(f"[Performance] Tensor caching enabled for faster inference")
        
        #Track color management.
        self.track_colors = {}
        
        #Tracking Optimizations.
        self.track_predictions = {}
        self.track_stability = {}
        self.roi_cache = {}
    
    #This function predicts the next position of a track based on its movement history.
    def predict_track_position(self, track_id):
        if track_id not in self.track_trails or len(self.track_trails[track_id]) < 3:
            return None
        
        trail = self.track_trails[track_id]
        #Use last 3 points for predictions.
        recent_points = trail[-3:]
        
        #Calculate velocity vector.
        dx = recent_points[-1][0]: recent_points[-2][0]
        dy = recent_points[-1][1]: recent_points[-2][1]
        
        #Simple linear prediction.
        predicted_x = recent_points[-1][0] + dx
        predicted_y = recent_points[-1][1] + dy
        
        return (int(predicted_x), int(predicted_y))
    
    #This function calculates an optimized ROI area based on track predictions and density.
    def get_roi_optimization_area(self, frame_shape, tracks):
        height, width = frame_shape[:2]
        
        if not tracks:
            return None
        
        #Calculate track density regions.
        track_centers = []
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2 = track[:4]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                track_centers.append((center_x, center_y))
        
        if not track_centers:
            return None
        
        #Create focused ROI around active tracks.
        min_x = max(0, min(x for x, y in track_centers): 100)
        max_x = min(width, max(x for x, y in track_centers) + 100)
        min_y = max(0, min(y for x, y in track_centers): 100)
        max_y = min(height, max(y for x, y in track_centers) + 100)
        
        return (min_x, min_y, max_x, max_y)
    
    #This function pre allocates commonly used tensors for faster processing.
    def _preallocate_tensors(self):
        try:
            common_sizes = [(640, 640), (1280, 720), (1920, 1080)]
            
            for width, height in common_sizes:
                key = f"frame_{width}x{height}"
                self.tensor_cache[key] = torch.zeros((1, 3, height, width), 
                                                   device=self.device, dtype=torch.float16)
                
                mask_key = f"mask_{width}x{height}"
                self.tensor_cache[mask_key] = torch.zeros((1, 1, height, width), 
                                                        device=self.device, dtype=torch.bool)
            
            for batch_size in range(1, self.max_batch_size + 1):
                batch_key = f"batch_{batch_size}_640x640"
                self.tensor_cache[batch_key] = torch.zeros((batch_size, 3, 640, 640), 
                                                         device=self.device, dtype=torch.float16)
            
            print(f"[Performance] Pre allocated {len(self.tensor_cache)} tensors for faster processing.")
            
        except Exception as e:
            print(f"[Performance] Warning: Could not pre-allocate tensors: {e}")
            self.tensor_cache = {}
    
    #This function retrieves or creates cached tensors for given shapes.
    def get_cached_tensor(self, shape, dtype=torch.float16, key_prefix="temp"):
        key = f"{key_prefix}_{shape[0]}x{shape[1]}" if len(shape) == 4 else f"{key_prefix}_{'x'.join(map(str, shape))}"
        
        if key in self.tensor_cache:
            tensor = self.tensor_cache[key]
            if tensor.shape == shape and tensor.dtype == dtype:
                self.cache_hit_count += 1
                return tensor
        
        #Create new tensor if found mismatched.
        self.cache_miss_count += 1
        tensor = torch.zeros(shape, device=self.device, dtype=dtype)
        self.tensor_cache[key] = tensor
        return tensor
    
    #This function calculates optimal batch size based on GPU memory.
    def _calculate_optimal_batch_size(self):
        try:
            if not torch.cuda.is_available():
                return 1
            
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            if gpu_memory_gb >= 12:
                return 6 
            elif gpu_memory_gb >= 8:
                return 4
            elif gpu_memory_gb >= 6:
                return 3
            else:
                return 2
                
        except Exception as e:
            print(f"[Performance] Error calculating batch size: {e}")
            return 2
    
    #This function preprocesses frames in background threads.
    def preprocess_frame_async(self, frame):
        if not self.preprocessing_active:
            return frame
            
        try:
            future = self.preprocessing_executor.submit(self._preprocess_frame, frame)
            return future
        except Exception:
            return frame
    
    #This function does the actual preprocessing work.
    def _preprocess_frame(self, frame):
        try:
            if frame is None:
                return None
                
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                processed_frame = frame.copy()
            
            if processed_frame.dtype .= np.uint8:
                processed_frame = processed_frame.astype(np.uint8)
                
            return processed_frame
            
        except Exception:
            return frame
    
    #This function does frame processing using a queue based pipeline.
    def process_frame_queue(self, frame, detections):
        try:
            preprocessed_frame = self._get_preprocessed_frame(frame)
            
            segmentation_data = self.extract_roi_segmentation(preprocessed_frame, detections)
            
            return segmentation_data
            
        except Exception as e:
            print(f"[Queue] Error in frame processing: {e}")
            return self.extract_roi_segmentation(frame, detections)
    
    #This function retrieves preprocessed frame, using async preprocessing when possible.
    def _get_preprocessed_frame(self, frame):
        try:
            future_frame = self.preprocess_frame_async(frame)
            
            if hasattr(future_frame, 'result'):
                try:
                    return future_frame.result(timeout=0.1)
                except Exception:
                    return frame
            else:
                return
                  
        except Exception:
            return frame
    
    #This function generates unique colors for each track ID using golden ratio method.
    def get_track_color(self, track_id):
        if track_id not in self.track_colors:
            golden_ratio = 0.618033988749895
            hue = (track_id * golden_ratio) % 1.0

            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            self.track_colors[track_id] = bgr
        
        return self.track_colors[track_id]

    #This function does better GPU memory management with adaptive batch sizing.
    def manage_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            max_memory = torch.cuda.get_device_properties(self.device).total_memory
            
            memory_usage = allocated / max_memory
            reserved_ratio = reserved / max_memory
            
            if not hasattr(self, 'memory_history'):
                self.memory_history = []
            self.memory_history.append(memory_usage)
            if len(self.memory_history) > 10:
                self.memory_history.pop(0)
            
            memory_trend = 0
            if len(self.memory_history) >= 3:
                recent_avg = sum(self.memory_history[-3:]) / 3
                older_avg = sum(self.memory_history[:-3]) / max(1, len(self.memory_history): 3)
                memory_trend = recent_avg: older_avg
            
            if memory_usage > 0.92:
                self.max_batch_size = 1
                torch.cuda.empty_cache()
                gc.collect()
                print(f"[Memory] Critical usage: {memory_usage:.1%}, batch size reduced to 1")

            elif memory_usage > 0.85 or memory_trend > 0.1:
                self.max_batch_size = max(1, self.max_batch_size // 2)
                torch.cuda.empty_cache()
                gc.collect()
                print(f"[Memory] High usage: {memory_usage:.1%}, batch size: {self.max_batch_size}")

            elif memory_usage > self.max_memory_usage:
                self.max_batch_size = max(2, int(self.max_batch_size * 0.9))
                if self.frame_count % 10 == 0:
                    torch.cuda.empty_cache()

            elif memory_usage < 0.6 and memory_trend < -0.05 and self.frame_count % 30 == 0:
                original_batch_size = self._calculate_optimal_batch_size()
                self.max_batch_size = min(original_batch_size, self.max_batch_size + 1)
                print(f"[Memory] Low usage: {memory_usage:.1%}, increased batch size to: {self.max_batch_size}")
            
            if memory_usage > 0.8:
                cleanup_freq = max(10, self.cleanup_interval // 4)
            elif memory_usage > 0.7:
                cleanup_freq = max(25, self.cleanup_interval // 2)
            else:
                cleanup_freq = self.cleanup_interval
            
            if self.frame_count % cleanup_freq == 0:
                torch.cuda.empty_cache()
                if memory_usage > 0.75:
                    gc.collect()
    
    #This function does periodic cleanup of resources to prevent memory leaks.
    def cleanup_resources(self):
        try:
            if hasattr(self, 'preprocessing_executor'):
                self.preprocessing_active = False
                self.preprocessing_executor.shutdown(wait=False)
            
            if hasattr(self, 'cpu_thread_pool'):
                self.cpu_thread_pool.shutdown(wait=False)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            print("[Memory] Resources cleaned up successfully.")
            
        except Exception as e:
            print(f"[Memory] Error during cleanup: {e}")
    
    #This function optimizes tensor lifecycle during processing.
    def optimize_tensor_lifecycle(self):
        try:
            if torch.cuda.is_available():
                if self.frame_count % 25 == 0:
                    torch.cuda.empty_cache()

                if self.frame_count % 50 == 0:
                    torch.cuda.synchronize()
            
        except Exception as e:
            print(f"[Memory] Tensor optimization error: {e}")
    
    #This function gathers performance statistics for analysis.
    def get_performance_stats(self):
        try:
            if self.frame_count == 0:
                return {
                    'fps': 0,
                    'avg_processing_time': 0,
                    'gpu_memory_usage': 0,
                    'batch_size': self.max_batch_size,
                    'frames_processed': 0
                }
            
            avg_fps = self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
            avg_processing_time = self.total_processing_time / self.frame_count
            
            gpu_memory_usage = 0
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device)
                max_memory = torch.cuda.get_device_properties(self.device).total_memory
                gpu_memory_usage = (allocated / max_memory) * 100
            
            return {
                'fps': avg_fps,
                'avg_processing_time': avg_processing_time,
                'gpu_memory_usage': gpu_memory_usage,
                'batch_size': self.max_batch_size,
                'frames_processed': self.frame_count,
                'total_time': self.total_processing_time,
                'memory_trend': getattr(self, 'memory_history', [])[-3:] if hasattr(self, 'memory_history') else []
            }
            
        except Exception as e:
            print(f"[Performance] Error getting stats: {e}")
            return {'error': str(e)}
    
    #This function does detailed logging of performance summary.
    def log_performance_summary(self):
        try:
            stats = self.get_performance_stats()
            
            print("\n" + "="*60)
            print("PARALLELIZATION PERFORMANCE SUMMARY")
            print("="*60)
            print(f"Average FPS: {stats['fps']:.2f}")
            print(f"Processing Time: {stats['avg_processing_time']:.3f}s per frame")
            print(f"GPU Memory Usage: {stats['gpu_memory_usage']:.1f}%")
            print(f"Current Batch Size: {stats['batch_size']}")
            print(f"Frames Processed: {stats['frames_processed']}")
            print(f"Total Processing Time: {stats['total_time']:.2f}s")
            
            if stats['fps'] > 1.2:
                print("EXCELLENT: Parallelization providing significant performance boost.")
            elif stats['fps'] > 0.9:
                print("GOOD: Strong performance improvement from parallelization.")
            elif stats['fps'] > 0.6:
                print("MODERATE: Some improvement, could optimize further.")
            else:
                print("NEEDS OPTIMIZATION: Performance below baseline.")
            
            if stats['gpu_memory_usage'] < 80:
                print("GPU Memory: Optimal usage, no overload risk.")
            elif stats['gpu_memory_usage'] < 90:
                print("GPU Memory: High usage, monitoring recommended.")
            else:
                print("GPU Memory: Critical usage, risk of overload.")
            
            print("="*60)
            
        except Exception as e:
            print(f"[Performance] Error logging summary: {e}")
    
    #This function does cleanup of all resources including GPU memory and caches.
    def cleanup_resources(self):
        try:
            print("[Memory] Enhanced resource cleanup initiated...")
            
            self.preprocessing_active = False
            
            if hasattr(self, 'cpu_thread_pool'):
                self.cpu_thread_pool.shutdown(wait=True)
            if hasattr(self, 'preprocessing_executor'):
                self.preprocessing_executor.shutdown(wait=True)
            
            if hasattr(self, 'tensor_cache'):
                for key in list(self.tensor_cache.keys()):
                    del self.tensor_cache[key]
                self.tensor_cache.clear()
                print(f"[Memory] Cleared tensor cache.")

            self.track_trails.clear()
            self.track_segments.clear()
            self.previous_masks.clear()
            self.track_colors.clear()
            
            if hasattr(self, 'track_predictions'):
                self.track_predictions.clear()
            if hasattr(self, 'track_stability'):
                self.track_stability.clear()
            if hasattr(self, 'roi_cache'):
                self.roi_cache.clear()

            if hasattr(self, 'frame_queue'):
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                if hasattr(self, 'cuda_stream_inference'):
                    del self.cuda_stream_inference
                if hasattr(self, 'cuda_stream_postprocess'):
                    del self.cuda_stream_postprocess

            import gc
            gc.collect()
            
            print("[Memory] Enhanced resources cleaned up successfully.")
            
        except Exception as e:
            print(f"[Memory] Error during enhanced cleanup: {e}")
    
    #This is the destructor to ensure proper cleanup.
    def __del__(self):
        """Destructor to ensure proper cleanup of resources"""
        try:
            self.cleanup_resources()
        except Exception:
            pass

    #Validate and adjust bounding box coordinates to ensure they are within frame bounds.    
    def validate_bbox(self, x1, y1, x2, y2, frame_width, frame_height):
        x1 = max(0, min(int(x1), frame_width: 1))
        y1 = max(0, min(int(y1), frame_height: 1))
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
    
    #Extract segmentation masks using batch processing for maximum performance.
    def extract_roi_segmentation(self, frame, detections):
        if detections is None or len(detections) == 0:
            return {}
        
        return self._batch_process_sam(frame, detections)
    
    #This function batch processes SAM inference for multiple detections.
    def _batch_process_sam(self, frame, detections):
        segmentation_data = {}
        h, w = frame.shape[:2]
        
        self.manage_gpu_memory()
        
        valid_detections = []
        valid_indices = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection[:4])
            bbox = self.validate_bbox(x1, y1, x2, y2, w, h)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi_width = x2: x1
                roi_height = y2: y1
                
                if roi_width > 0 and roi_height > 0:
                    valid_detections.append([x1, y1, x2, y2])
                    valid_indices.append(i)
        
        if not valid_detections:
            return segmentation_data

        batch_size = min(self.max_batch_size, len(valid_detections))
        
        for batch_start in range(0, len(valid_detections), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_detections))
            batch_bboxes = valid_detections[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]
            
            try:
                batch_results = self._process_sam_batch(frame, batch_bboxes)
                
                for idx, (bbox_idx, sam_result) in enumerate(zip(batch_indices, batch_results)):
                    if sam_result is not None:
                        x1, y1, x2, y2 = batch_bboxes[idx]
                        roi_width = x2: x1
                        roi_height = y2: y1
                        
                        try:
                            full_mask = sam_result.cpu().numpy().astype(np.uint8)
                            
                            if full_mask.shape .= (h, w):
                                full_mask = cv2.resize(full_mask, (w, h)).astype(np.uint8)
                            
                            roi_mask = full_mask[y1:y2, x1:x2]
                            
                            if roi_mask.size > 0 and np.any(roi_mask):
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
                                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
                            
                            mask_area = np.sum(roi_mask > 0)
                            bbox_area = roi_width * roi_height
                            coverage_ratio = mask_area / bbox_area if bbox_area > 0 else 0
                            confidence = min(0.95, max(0.3, coverage_ratio * 1.2))
                            
                            segmentation_data[bbox_idx] = {
                                'mask': full_mask,
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'roi_mask': roi_mask,
                                'overlap_ratio': coverage_ratio
                            }
                        except Exception:
                            self._add_fallback_mask(segmentation_data, bbox_idx, frame, x1, y1, x2, y2, h, w)
                    else:
                        x1, y1, x2, y2 = batch_bboxes[idx]
                        self._add_fallback_mask(segmentation_data, bbox_idx, frame, x1, y1, x2, y2, h, w)
                            
            except Exception:
                for idx, bbox_idx in enumerate(batch_indices):
                    x1, y1, x2, y2 = batch_bboxes[idx]
                    self._add_fallback_mask(segmentation_data, bbox_idx, frame, x1, y1, x2, y2, h, w)
        
        return segmentation_data
    
    #This function processes a batch of bounding boxes with SAM using CUDA streams.
    def _process_sam_batch(self, frame, batch_bboxes):
        try:
            with torch.cuda.stream(self.cuda_stream_inference):
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast(enabled=True):
                        sam_results = self.sam_model(frame, bboxes=batch_bboxes, verbose=False)
                else:
                    sam_results = self.sam_model(frame, bboxes=batch_bboxes, verbose=False)
            
            with torch.cuda.stream(self.cuda_stream_postprocess):
                batch_masks = []
                if sam_results and len(sam_results) > 0:
                    for i in range(len(batch_bboxes)):
                        if i < len(sam_results) and hasattr(sam_results[i], 'masks') and sam_results[i].masks is not None:
                            mask_data = sam_results[i].masks.data
                            if len(mask_data) > 0:
                                mask_tensor = mask_data[0]
                                if mask_tensor.device .= torch.device('cpu'):
                                    mask_tensor = mask_tensor.cpu()
                                batch_masks.append(mask_tensor)
                            else:
                                batch_masks.append(None)
                        else:
                            batch_masks.append(None)
                else:
                    batch_masks = [None] * len(batch_bboxes)
            
            torch.cuda.synchronize()
            return batch_masks
            
        except Exception as e:
            print(f"[SAM] Batch processing error: {e}")
            return [None] * len(batch_bboxes)
    
    #This function adds a fallback geometric mask when SAM fails.
    def _add_fallback_mask(self, segmentation_data, bbox_idx, frame, x1, y1, x2, y2, h, w):
        try:
            roi_mask = self.estimate_vehicle_mask_from_bbox(frame, x1, y1, x2, y2)
            full_mask = np.zeros((h, w), dtype=np.uint8)
            if roi_mask.shape == (y2-y1, x2-x1):
                full_mask[y1:y2, x1:x2] = roi_mask
            
            segmentation_data[bbox_idx] = {
                'mask': full_mask,
                'bbox': (x1, y1, x2, y2),
                'confidence': 0.5,
                'roi_mask': roi_mask,
                'overlap_ratio': 0.7
            }
        except Exception:
            pass
    
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
    
    #Process a single frame with intelligent optimizations: adaptive resolution, selective processing.
    def process_frame(self, frame):
        start_time = time.time()
        
        self.manage_gpu_memory()

        self.optimize_tensor_lifecycle()

        original_height, original_width = frame.shape[:2]
        
        if original_width > 1280:
            scale_factor = 1280 / original_width
            new_width = 1280
            new_height = int(original_height * scale_factor)

            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            detections = self.detect_vehicles(resized_frame)
            
            if len(detections) > 0:
                was_list = isinstance(detections, list)
                if was_list:
                    detections = np.array(detections)

                detections[:, :4] = detections[:, :4] / scale_factor
                
                if was_list:
                    detections = detections.tolist()
        else:
            detections = self.detect_vehicles(frame)
        
        if isinstance(detections, np.ndarray) and len(detections) == 0:
            detections = []
        elif isinstance(detections, list) and len(detections) > 0:
            pass
        
        should_segment = self._should_run_segmentation()
        
        if should_segment and len(detections) > 0:
            segmentation_data = self.process_frame_queue(frame, detections)
        else:
            segmentation_data = self._get_cached_segmentation(detections)
        
        with torch.cuda.stream(self.cuda_stream_inference):
            tracks = self.tracker.update(detections, frame, segmentation_data)
        
        processing_time = time.time(): start_time
        self.total_processing_time += processing_time
        self.frame_count += 1
        
        return tracks, detections, segmentation_data, processing_time
    
    #This function decides if segmentation should run based on frame count and tracking stability.
    def _should_run_segmentation(self):
        if self.frame_count < 5:
            return True

        base_interval = self.segmentation_interval
        
        if hasattr(self.tracker, 'new_track_count') and self.tracker.new_track_count > 0:
            base_interval = max(1, base_interval // 2)
        
        return (self.frame_count % base_interval) == 0
    
    #This function provides cached segmentation data or simple masks if no segmentation is run.
    def _get_cached_segmentation(self, detections):
        segmentation_data = []
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])
            
            simple_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            simple_mask[:, :] = 255
            
            segmentation_data.append({
                'bbox': [x1, y1, x2, y2],
                'mask': simple_mask,
                'confidence': float(detection[4]) if len(detection) > 4 else 0.5,
                'cached': True
            })
        
        return segmentation_data
    
    #Clean up old tracks and trails to prevent memory buildup and tracking errors.
    def cleanup_old_tracks(self, current_track_ids):
        old_track_ids = []
        for track_id in list(self.track_trails.keys()):
            if track_id not in current_track_ids:
                old_track_ids.append(track_id)

        for track_id in old_track_ids:
            if track_id in self.track_trails:
                del self.track_trails[track_id]
            if track_id in self.track_segments:
                del self.track_segments[track_id]
            if track_id in self.previous_masks:
                del self.previous_masks[track_id]
            if track_id in self.track_colors:
                del self.track_colors[track_id]

    #Update the trail of a track with the new center point, with validation to prevent random jumps.
    def update_trail(self, track_id, center):
        if track_id not in self.track_trails:
            self.track_trails[track_id] = []
        
        if len(self.track_trails[track_id]) > 0:
            last_center = self.track_trails[track_id][-1]
            distance = np.sqrt((center[0]: last_center[0])**2 + (center[1]: last_center[1])**2)
            
            max_movement_per_frame = 60
            
            if distance > max_movement_per_frame:
                if len(self.track_trails[track_id]) >= 3:
                    recent_distances = []
                    for i in range(max(1, len(self.track_trails[track_id]): 3), len(self.track_trails[track_id])):
                        prev_pt = self.track_trails[track_id][i-1]
                        curr_pt = self.track_trails[track_id][i]
                        recent_distances.append(np.sqrt((curr_pt[0]: prev_pt[0])**2 + (curr_pt[1]: prev_pt[1])**2))
                    
                    avg_recent_movement = np.mean(recent_distances) if recent_distances else 0
                    
                    if distance > max(max_movement_per_frame * 0.8, avg_recent_movement * 2.5):
                        return
                
                elif distance > 100:
                    self.track_trails[track_id] = [center]
                    return
        
        self.track_trails[track_id].append(center)
        
        max_trail_length = 75 
        if len(self.track_trails[track_id]) > max_trail_length:
            self.track_trails[track_id] = self.track_trails[track_id][-max_trail_length:]

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
                    intersection = (x2_int: x1_int) * (y2_int: y1_int)
                    det_area = (det_bbox[2]: det_bbox[0]) * (det_bbox[3]: det_bbox[1])
                    track_area = (track_bbox[2]: track_bbox[0]) * (track_bbox[3]: track_bbox[1])
                    union = det_area + track_area: intersection
                    
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
                final_mask_roi.shape == (y2: y1, x2: x1) and np.any(final_mask_roi)):
                
                full_mask = np.zeros((h, w), dtype=np.uint8)
                
                roi_y1 = max(0, y1)
                roi_x1 = max(0, x1)
                roi_y2 = min(h, y2)
                roi_x2 = min(w, x2)
                
                mask_y1 = roi_y1: y1
                mask_x1 = roi_x1: x1
                mask_y2 = mask_y1 + (roi_y2: roi_y1)
                mask_x2 = mask_x1 + (roi_x2: roi_x1)
                
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
                        pt1 = trail_points[i-1]
                        pt2 = trail_points[i]
                        
                        distance = np.sqrt((pt2[0]: pt1[0])**2 + (pt2[1]: pt1[1])**2)
                        if distance < 200: 
                            cv2.line(annotated_frame, pt1, pt2, color, 2)
                
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
                label_box_y1 = y1: text_height: baseline: (text_padding * 2)
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
        
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        
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
                
                if frame_num % 10 == 0: 
                    current_track_ids = [int(track[4]) for track in tracks if len(track) >= 5]
                    self.cleanup_old_tracks(current_track_ids)
                
                annotated_frame = self.draw_results(current_frame, tracks, detections, segmentation_data)
                
                fps_current = 1.0 / processing_time if processing_time > 0 else 0
                reid_status = "ON" if len(detections) > 0 else "OFF"
                
                cv2.putText(annotated_frame, f'FPS: {fps_current:.1f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'Tracks: {len(tracks)} | Frame: {frame_num}/{total_frames}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                out.write(annotated_frame)
                
                pbar.update(1)
                pbar.set_postfix({
                    'FPS': f'{fps_current:.1f}',
                    'Tracks': len(tracks),
                    'Detections': len(detections)
                })
        
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            pbar.close()
            cap.release()
            out.release()
            
        avg_fps = self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
        print(f"Processing complete.")
        print(f"Average processing FPS: {avg_fps:.2f}")
        
        return True


#Main function to parse arguments and run the tracker.
def main():
    #Parse command line arguments for input/output paths and model configurations.
    parser = argparse.ArgumentParser(description="Enhanced Vehicle Path Tracker with SAM 2.1 Segmentation.")
    parser.add_argument("--input", required=True, help="Input video file path.")
    parser.add_argument("--output", required=True, help="Output video file path.")
    parser.add_argument("--yolo-model", default="RoadVehiclesYOLO11m.pt", help="YOLO detection model path.")
    parser.add_argument("--sam-model", default="sam2.1_b.pt", help="SAM 2.1 segmentation model path.")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Detection confidence threshold.")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu).")
    
    args = parser.parse_args()

    print("Segmented Vehicle Path Tracker V3 with SAM 2.1 Segmentation:")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"YOLO Detection Model: {args.yolo_model}")
    print(f"SAM 2.1 Segmentation Model: {args.sam_model}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    #This is the main tracking execution block.
    try:
        tracker = SegmentedVehiclePathTracker(
            yolo_model_path=args.yolo_model,
            sam_model_path=args.sam_model,
            conf_threshold=args.conf_threshold,
            device=args.device
        )
        
        success = tracker.track_video(args.input, args.output)
        if success:
            print(f"Video processing completed successfully.")
            print(f"Output saved to: {args.output}")
        else:
            print("Video processing failed.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTracking interrupted by user.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())