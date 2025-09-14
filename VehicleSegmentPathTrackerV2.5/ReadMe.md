# Vehicle Segment Path Tracker V2.5:

A real time vehicle detection, segmentation, and tracking system that combines YOLOv11 detection and segmentation models with DeepSORT tracking enhanced by pretrained MobileNetV2 feature extraction. This system provides accurate multi object tracking with visual path visualization and segmentation overlays.

![ForeignRoundaboutsStackedSegmentedDeepSORT](visuals/ForeignRoundaboutsStackedSegmentedDeepSORT.gif)

## Features:

- **Multi Model Detection**: Utilizes both YOLOv11 detection and YOLOv11 segmentation models for vehicle identification.
- **Advanced Tracking**: DeepSORT tracker with MobileNetV2-based feature extraction for reliable multi object tracking.
- **Segmentation Overlay**: Real time vehicle segmentation masks with morphological post processing.
- **Path Visualization**: Dynamic trail rendering showing vehicle movement paths.
- **Adaptive Processing**: Dynamic confidence thresholds and IoU adjustments based on detection size.
- **Optimized Performance**: CUDA acceleration with efficient memory management.

## Architecture Overview:

The system implements a stacked detection and segmentation approach:

1. **Primary Detection**: YOLOv11 model detects vehicle bounding boxes
2. **Segmentation Refinement**: YOLOv11-seg extracts precise vehicle masks within detected regions
3. **Feature Extraction**: MobileNetV2 generates appearance features for tracking
4. **Tracking**: DeepSORT associates detections across frames using motion and appearance models
5. **Visualization**: Renders tracking results with segmentation overlays and path trails

## Models and Algorithms:

### Detection Models:
- **YOLOv11**: Primary vehicle detection model.
- **YOLOv11-seg**: Segmentation model for precise vehicle boundaries.
- **Vehicle Classes**: Supports detection of cars, trucks, buses, motorcycles, and other vehicle types.

### Tracking Algorithm:
- **DeepSORT**: Multi object tracking with Kalman filter for motion prediction.
- **Feature Extractor**: MobileNetV2 pretrained on ImageNet for appearance based reidentification.
- **ReID Model**: VeRiWild R50 IBN model for enhanced vehicle reidentification (requires newFastReID).

### Configuration Parameters:
- **Confidence Threshold**: 0.35 (adjustable).
- **IoU Threshold**: Adaptive based on detection size (0.2-0.35).
- **Track Buffer**: 100 frames for handling temporary occlusions.
- **Minimum Track Length**: 3 consecutive frames for track initialization.

## Dependencies:

### Core Requirements:
```
torch>=1.9.0
torchvision>=0.10.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
supervision>=0.3.0
Pillow>=8.0.0
PyYAML>=5.4.0
```

### Tracking Library
```
trackers  #DeepSORT implementation with feature extraction support.
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/WhiteMetagross/ProjectIAV/tree/main/VehicleSegmentPathTrackerV2.5.git
cd VehicleSegmentPathTrackerV2.5
```

2. **Install Python dependencies**:
```bash
pip install torch torchvision ultralytics opencv-python numpy supervision Pillow PyYAML
```

3. **Install tracking library**:
```bash
pip install trackers
```

4. **Download model weights**:
   - Place your trained YOLOv11 detection model at the specified path.
   - Download YOLOv11-seg model: `yolo11m-seg.pt`.

## Usage:

### Basic Usage:
```bash
python VehicleSegmentPathTrackerv2_5.py --input input_video.mp4 --output output_video.mp4
```

### Advanced Usage with Custom Models:
```bash
python VehicleSegmentPathTrackerv2_5.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --yolo-model path/to/custom_yolo_model.pt \
    --segmentation-model path/to/segmentation_model.pt \
    --conf-threshold 0.4 \
    --device cuda
```

### Command Line Arguments:
- `--input`: Input video file path (required)
- `--output`: Output video file path (required)
- `--yolo-model`: Path to YOLOv11 detection model (optional)
- `--segmentation-model`: Path to YOLOv11 segmentation model (optional)
- `--conf-threshold`: Detection confidence threshold (default: 0.3)
- `--device`: Processing device - 'cuda' or 'cpu' (default: 'cuda')

### Programmatic Usage:
```python
from VehicleSegmentPathTrackerv2_5 import SegmentedVehiclePathTracker

#Initialize tracker:
tracker = SegmentedVehiclePathTracker(
    yolo_model_path="path/to/yolo_model.pt",
    segmentation_model_path="path/to/yolo11m-seg.pt",
    conf_threshold=0.35,
    device='cuda'
)

#Process video:
success = tracker.track_video("input.mp4", "output.mp4")
```

## Algorithm Details:

### Stacked Detection and Segmentation:
1. **Primary Detection**: YOLOv11 identifies vehicle bounding boxes.
2. **Region of Interest (ROI) Extraction**: Expands detection boxes with adaptive padding.
3. **Segmentation**: YOLOv11-seg processes Region of Interest (ROI) for precise masks.
4. **Mask Refinement**: Morphological operations improve mask quality.
5. **Overlap Scoring**: Combines confidence and spatial overlap for best mask selection.

### Adaptive Thresholding:
The system dynamically adjusts detection parameters based on region size:
- **Small regions** (<10k pixels): Lower thresholds for better sensitivity
- **Medium regions** (10k-50k pixels): Balanced thresholds
- **Large regions** (>50k pixels): Higher thresholds to reduce false positives

### Track Management:
- **Kalman Filtering**: Predicts vehicle motion between frames
- **Appearance Modeling**: MobileNetV2 features for robust reidentification
- **Track Lifecycle**: Handles track initialization, maintenance, and deletion
- **Trail Visualization**: Maintains historical path data for trajectory rendering

## Performance Optimization:

- **CUDA Acceleration**: GPU processing for real time performance
- **Memory Management**: Efficient handling of video streams and model weights
- **Batch Processing**: Optimized detection and segmentation workflows
- **Adaptive Quality**: Dynamic parameter adjustment based on detection characteristics

## Output Visualization:

The system provides:
- **Bounding Boxes**: Color-coded vehicle detections
- **Segmentation Masks**: Semi-transparent vehicle outlines
- **Track IDs**: Persistent vehicle identification numbers
- **Movement Trails**: Historical path visualization
- **Status Information**: Frame count and processing statistics

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU processing.
2. **Model loading errors**: Verify model file paths and compatibility.
3. **Tracking instability**: Adjust confidence thresholds or IoU parameters.
4. **Performance issues**: Ensure CUDA drivers are properly installed.

### Performance Tips:
- Use GPU acceleration when available.
- Optimize confidence thresholds for your specific use case.
- Consider input video resolution vs. processing speed trade offs.
- Monitor memory usage for long video sequences.

## Acknowledgments:

- **YOLOv11**: Ultralytics for object detection and segmentation models.
- **DeepSORT**: Multi object tracking algorithm implementation.
- **newFastReID**: Enhanced vehicle reidentification capabilities.
- **MobileNetV2**: Feature extraction backbone for tracking.

## Citation:

If you use this work in your research, please cite:
```bibtex
@software{vehicle_segment_path_tracker,
  title={Vehicle Segment Path Tracker V2.5},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/VehicleSegmentPathTrackerV2.5}
}

```

