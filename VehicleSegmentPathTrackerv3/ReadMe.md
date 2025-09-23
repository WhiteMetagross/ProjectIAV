# Vehicle Segementation and Path Tracker V3:

A vehicle tracking system that combinesa fine tuned YOLO11m based object detection, Meta's SAM 2.1 segmentation models, and a custom BoTSORT tracking with FastReID feature extraction. This system achieves performance with 4,700% speed improvements over baseline implementations through advanced GPU optimizations and good processing strategies.

## Visual Demonstration:

The system provides vehicle tracking with relatively accurate segmentation overlays, persistent track identification, and path visualization. The following demonstration shows the tracker processing aerial roundabout footage with multiple vehicles:

**Vehicle Tracking Demo:**

![Vehicle Tracking Demo](visuals\ForeignRoundaboutSAMSegemtedTracker.gif)

This visualization demonstrates:
- Multi vehicle detection and tracking in complex traffic scenarios.
- relatively accurate segmentation masks with semi transparent overlays.
- Persistent track ID assignment with color coded identification.
- path trail visualization showing vehicle trajectories
- Processing of challenging roundabout navigation with vehicle interactions.
- Smooth tracking performance at 10+ FPS processing speeds.

## Documentation Structure:

This project includes documentation to support development, deployment, and maintenance:

**README.md:** This file provides complete system overview, installation instructions, usage guidelines, and performance analysis.

**CODEBASE_INDEX.md:** Detailed technical documentation containing file structure analysis, implementation details, optimization techniques, and architectural design patterns.

**requirements.txt:** Complete dependency specification with version constraints and installation instructions.

**veriwild_r50_ibn_config.yml:** FastReID model configuration for vehicle reidentification.

## Performance Highlights:

- **High Performance**: 10+ FPS processing with 4,700% improvement over 0.5 FPS baseline.
- **GPU Acceleration**: Advanced CUDA optimizations with tensor caching and memory management.
- **SAM Models**: Support for SAM 2.1 Base and Small with 50% performance difference, bwtween them.
- **Processing**: Optimized for RTX 4060 and higher with good resource management.
- **Code Quality**: Production ready system with error handling and monitoring.

## Key Features:

### Detection and Segmentation:
- **Custom YOLO11m Detection**: Custom fine tuned model for vehicle detection, from a top down aerial view.
- **SAM 2.1 Segmentation**: Support for both Base (accuracy) and Small (speed) models.
- **Bounding Box Prompts**: Direct integration eliminating ROI preprocessing overhead.
- **Good Processing**: Adaptive resolution scaling and selective segmentation intervals.

### Advanced Tracking:
- **BoTSORT Implementation**: Multi object tracking with Kalman filtering.
- **FastReID Integration**: VeriWild R50-IBN model for appearance based reidentification.
- **Segmentation Mask Enhanced Tracking**: Segmentation masks improve tracking accuracy.
- **Trail Visualization**: Path rendering with validated trajectories.

### Performance Optimizations:
- **PyTorch Native Acceleration**: Flash SDP, Memory Efficient SDP, and Math SDP backends.
- **CUDA Streams**: Parallel inference and post processing pipelines.
- **Tensor Caching**: Pre allocated tensors for common operations.
- **Mixed Precision**: Automatic FP16 processing when supported.
- **good Scheduling**: Adaptive frame processing with stability-based segmentation.

## System Architecture:

### Core Pipeline:
1. **Adaptive Frame Processing**: Good resolution scaling for large frames.
2. **YOLO11m Detection**: Vehicle bounding box identification.
3. **SAM 2.1 Segmentation**: Relatively accurate mask generation using bbox prompts.
4. **Feature Extraction**: FastReID appearance features for tracking.
5. **BoTSORT Tracking**: Multi object association and trajectory management.
6. **Visualization**: rendering with segmentation overlays.

### Model Specifications:

#### Detection Model:
- **Architecture**: YOLO11m (ultralytics).
- **Training**: Fine tuned on aerial vehicle datasets.
- **Classes**: Cars, trucks, buses, motorcycles, and other vehicles.
- **Input Resolution**: Adaptive scaling (up to 1280px width).
- **Confidence Threshold**: 0.35 (configurable).

#### Segmentation Models:
- **SAM 2.1 Base**: High accuracy model (sam2.1_b.pt).
  - Processing time: Standard baseline.
  - Memory usage: ~6.7% GPU on RTX 4060.
  - Best for accuracy-critical applications.
- **SAM 2.1 Small**: High speed model (sam2.1_s.pt).
  - Processing time: 50% faster than Base model.
  - Lower memory footprint.
  - Optimal for applications.

#### Reidentification Model:
- **Architecture**: ResNet50 with IBN (Instance Batch Normalization).
- **Dataset**: VeriWild vehicle reidentification dataset.
- **Feature Dimension**: 2048 dimensional embeddings.
- **Framework**: newFastReID implementation.

### Performance Optimizations:

#### GPU Acceleration:
- **PyTorch Flash SDP**: Hardware accelerated attention mechanisms.
- **CUDA Streams**: Parallel GPU operations for inference and post processing.
- **Memory Management**: 95% GPU memory utilization with good cleanup.
- **Tensor Caching**: Pre allocated tensors reduce allocation overhead.

#### Processing:
- **Batch Processing**: Dynamic batch size adjustment based on GPU memory
- **Frame Scheduling**: Relative Segmentation intervals based on tracking stability.
- **Resolution Adaptation**: Automatic scaling for frames larger than 1280px.
- **Queue based Pipeline**: Background preprocessing with ThreadPoolExecutor.

#### Memory Management:
- **Adaptive Batch Sizing**: Adaptive adjustment based on memory usage trends.
- **Cleanup Strategies**: Frequency based cache clearing and garbage collection.
- **Resource Monitoring**: Continuous GPU memory usage tracking.
- **Error Recovery**: Fallback mechanisms for memory overflow situations.

## Requirements and Dependencies:

### System Requirements:
- **GPU**: NVIDIA RTX 4060 or higher with 8GB+ VRAM (RTX 3060 Ti minimum).
- **CUDA**: Version 12.1 or higher.
- **Python**: 3.8+.
- **Memory**: 8GB+ RAM recommended.
- **Storage**: 10GB+ for models and dependencies.

### Core Dependencies:

```
  PyTorch 2.3.0 with CUDA 12.1 support for deep learning framework.
  TorchVision 0.18.0 with CUDA acceleration for vision utilities.
  Ultralytics 8.3.0 for YOLO11m and SAM 2.1 integration.
  OpenCV 4.8.0 for computer vision operations.
  NumPy 1.24.0 for numerical computing.
  FastReID for ReID feature extraction.
  FilterPy 1.4.5 for Kalman filtering.
  SciPy 1.10.0 for scientific computing.
  TQDM 4.65.0 for progress bars.
  Pillow 10.0.0 for image processing.
  PyYAML 6.0 for configuration files.
  Matplotlib 3.6.0 for visualization utilities.
```

### Model Files:
- **YOLO11m**: Custom fine tuned detection model (user-provided).
- **SAM 2.1 Base**: `sam2.1_b.pt` (auto downloaded by ultralytics).
- **SAM 2.1 Small**: `sam2.1_s.pt` (auto downloaded, 50% faster).
- **FastReID**: VeriWild R50-IBN model (`veriwild_bot_R50-ibn.pth`).
- **Configuration**: `veriwild_r50_ibn_config.yml`.

## Installation:

### 1. Environment Setup:
Create conda environment with Python 3.10
Activate the vehicle-tracker environment
Install PyTorch with CUDA 12.1 support using the official PyTorch index.

### 2. Install Dependencies:
Clone the repository to your local system
Navigate to the VehicleSegmentPathTrackerv3 directory
Install all requirements from the requirements.txt file
Install FastReID from the GitHub source repository.

### 3. Model Setup:
SAM models are automatically downloaded on first use
Place your custom YOLO11m model in the project directory
Download VeriWild ReID model and configuration files.

## Usage:

### Command Line Interface:

Basic usage command structure:

- python VehicleSegmentPathTrackerv3.py --input video.mp4 --output tracked_video.mp4

- High performance mode with SAM 2.1 Small:

  python VehicleSegmentPathTrackerv3.py --input video.mp4 --output tracked_video.mp4 --sam-model sam2.1_s.pt --conf-threshold 0.4

- Full configuration with all parameters:

  python VehicleSegmentPathTrackerv3.py --input video.mp4 --output tracked_video.mp4 --yolo-model custom_yolo11m.pt --sam-model sam2.1_b.pt --conf-threshold 0.35 --device cuda

### Programmatic Usage:
- Import the SegmentedVehiclePathTracker class from VehicleSegmentPathTrackerv3.

- Initialize with high-performance configuration specifying model paths, SAM model selection, ReID model configuration, confidence threshold, and processing device.

- Process video with performance monitoring using the track_video method.

- Retrieve performance statistics including FPS, GPU memory usage, and processing metrics.

- Clean up resources using the cleanup_resources method.

## Configuration Options:

### Command Line Arguments:

- `--input`: Input video file path (required).
- `--output`: Output video file path (required).  
- `--yolo-model`: Path to YOLO11m detection model.
- `--sam-model`: SAM model choice (`sam2.1_b.pt` or `sam2.1_s.pt`).
- `--conf-threshold`: Detection confidence (0.1-1.0, default: 0.35).
- `--device`: Processing device (`cuda` or `cpu`).

## Technical Implementation Details:

### Optimization Techniques:

#### 1. PyTorch Native Acceleration:
Enabled optimizations include Flash Attention, Memory Efficient Attention, TensorFloat-32 precision, and cuDNN benchmark mode for optimal GPU performance.

#### 2. CUDA Stream Processing:
Parallel GPU operations utilize separate streams for SAM inference and post processing to maximize hardware utilization.

#### 3. Memory Management:
Adaptive batch sizing based on GPU memory monitoring with emergency reduction protocols and good cleanup strategies

#### 4. Frame Processing:
Adaptive resolution scaling automatically reduces large frame dimensions while scaling coordinates back to original resolution for improved processing speed.

### Architecture Components:

#### Detection Pipeline:
1. **Adaptive Scaling**: Large frames (>1280px) are goodly scaled.
2. **YOLO11m Inference**: Custom fine tuned model for vehicle detection.
3. **Coordinate Scaling**: Results scaled back to original resolution.

#### Segmentation Pipeline:
1. **Batch Processing**: Multiple detections processed simultaneously.
2. **SAM 2.1 Inference**: Bounding box prompts generate precise masks.
3. **Post Processing**: Morphological operations improve mask quality.
4. **Fallback Handling**: Geometric estimation when SAM fails.

#### Tracking Pipeline:
1. **Feature Extraction**: FastReID generates appearance embeddings.
2. **Motion Prediction**: Kalman filter predicts next positions.
3. **Data Association**: Hungarian algorithm matches detections to tracks.
4. **Track Management**: Lifecycle handling with hit streak validation.

### SAM Model Comparison:
| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| SAM 2.1 Base | Baseline | Highest | 6.7% GPU | Production accuracy |
| SAM 2.1 Small | 50% faster | High | Lower | applications |

### Processing Pipeline Performance:
1. **Frame Preprocessing**: ~2ms (parallel CPU threads).
2. **YOLO11m Detection**: ~8ms (GPU optimized).
3. **SAM 2.1 Segmentation**: ~15ms (Base) / ~7ms (Small).
4. **Feature Extraction**: ~5ms (FastReID).
5. **Tracking Association**: ~3ms (CPU optimized).
6. **Visualization**: ~7ms (GPU accelerated).

## Troubleshooting:

### Common Issues:

#### CUDA Out of Memory:
Symptoms include RuntimeError: CUDA out of memory errors during processing.

Solutions involve using SAM 2.1 Small model with sam2.1_s.pt parameter, reducing batch size in code or using CPU fallback, and monitoring GPU memory with nvidia-smi.

#### Performance Issues:
Symptoms include low FPS and high processing time measurements.

Solutions require verifying CUDA drivers installation, checking PyTorch CUDA availability, using appropriate SAM model for hardware, and monitoring CPU/GPU usage.

#### Model Loading Errors:
Symptoms include FileNotFoundError and model loading failures.

Solutions involve verifying model paths in configuration, checking internet connection for auto downloads, manually downloading SAM models from ultralytics, and validating FastReID installation.

#### Tracking Instability:
Symptoms include ID switches and poor track continuity.

Solutions require adjusting confidence threshold to 0.4, modifying IoU thresholds in botsort.py, checking video quality and resolution, and tuning ReID similarity thresholds.

### Performance Optimization Advice:

#### Hardware Optimization:
- Use RTX 4060 or higher for optimal performance.
- Ensure adequate cooling for sustained GPU loads.
- 16GB+ RAM recommended for large videos.
- NVMe SSD storage for faster I/O.

#### Software Configuration:
Configuration utilizes SAM 2.1 Small model (sam2.1_s.pt) as the fastest model option, higher confidence threshold of 0.4, and CUDA device acceleration.

Memory optimized configuration employs SAM 2.1 Small model (sam2.1_s.pt) for lower memory usage, balanced confidence threshold of 0.35, and CUDA device processing.

#### Video Processing Advice:
- Process at original resolution when possible.
- Use H.264 codec for input videos.
- Monitor performance with progress bars.
- Consider batch processing for multiple videos.

## Development and Extension:

### Adding Custom Models:
Custom YOLO model replacement involves specifying the yolo_model_path parameter with path to custom YOLO weights, maintaining other standard parameters.

Custom ReID model integration requires specifying reid_model_path with path to custom ReID weights and reid_config_path with path to custom configuration YAML file.

### Performance Monitoring:
Performance tracking provides FPS measurements, memory usage percentages, batch size monitoring, and performance statistics.

Detailed logging includes detailed performance summaries with processing time analysis, GPU utilization metrics, and optimization effectiveness reports.

### Custom Optimizations:
Processing interval modification allows adjusting detection_interval to run detection every frame and segmentation_interval to run segmentation every specified number of frames.

Memory management adjustment enables modifying max_memory_usage to utilize specified percentage of GPU memory and cleanup_interval to perform cleanup every specified number of frames.

## Research and Citations:

### Datasets and Training:
- **YOLO Training**: [Aerial Vehicle OBB Dataset](https://www.kaggle.com/datasets/redzapdos123/aerial-vehicle-obb-dataset).
- **ReID Training**: [VeriWild Dataset](https://github.com/PKU-IMRE/VERI-Wild) for vehicle reidentification.
- **Validation**: Custom aerial footage from various traffic scenarios.

### Algorithm References
SAM 2.1: Segment Anything in Images and Videos by Meta AI Research, 2024.

BoTSORT: Robust Associations Multi-Pedestrian Tracking, 2022.

FastReID: A Pytorch Toolbox for General Instance Reidentification, 2020.

### Performance Validation
- Tested on RTX 4060 Laptop GPU with 8GB VRAM.
- Validated with 1080p and 4K aerial vehicle footage.
- Benchmarked against baseline implementations.
- processing verified at 10+ FPS.

## License and Acknowledgments:

### Third Party Components
- **Ultralytics**: YOLO11m based model and SAM 2.1 integration.
- **Meta AI**: SAM 2.1 segmentation models.
- **newFastReID**: Enhanced vehicle reidentification.
- **PyTorch**: Deep learning framework with CUDA acceleration.
- **OpenCV**: Computer vision operations.

### Citation:
Vehicle Segment Path Tracker V3

Author: Mridankan Mandal.

Year: 2025

URL: [Repository URL]

Note: Achieves 4,700% performance improvement over baseline implementations.
