# Vehicle Segment Path Tracker V2:

A real time vehicle detection, segmentation, and tracking system that combines fine tuned YOLO11m detection and YOLO11m-OBB segmentation models with custom BoTSORT tracking enhanced by VeriWild R50 IBN ReID feature extraction. This system provides accurate multi object tracking with visual path visualization and segmentation overlays for aerial vehicle footage.

![ForeignRoundaboutsStackedSegmentedDeepSORT](visuals\ForeignRoundaboutStackedSegmentTrackerBoTSORT.gif)

## Features:

- **Multi Model Detection**: Utilizes fine tuned YOLO11m detection and YOLO11m-OBB segmentation models for aerial vehicle identification.
- **Advanced Tracking**: Custom BoTSORT tracker with VeriWild R50 IBN ReID based feature extraction for reliable multi object tracking.
- **Segmentation Overlay**: Real time vehicle segmentation masks with morphological post processing.
- **Path Visualization**: Dynamic trail rendering showing vehicle movement paths.
- **Adaptive Processing**: Dynamic confidence thresholds and IoU adjustments based on detection size.
- **Optimized Performance**: CUDA acceleration with efficient memory management.

## Architecture Overview:

The system implements a stacked detection and segmentation approach:

1. **Primary Detection**: Fine tuned YOLO11m model (trained on aerial vehicle OBB dataset) detects vehicle bounding boxes
2. **Segmentation Refinement**: YOLO11m-OBB extracts precise vehicle masks within detected regions
3. **Feature Extraction**: VeriWild R50 IBN ReID model generates appearance features for tracking through newFastReID
4. **Tracking**: Custom BoTSORT associates detections across frames using motion and appearance models
5. **Visualization**: Renders tracking results with segmentation overlays and path trails

## Models and Algorithms:

### Detection Models:
- **YOLO11m**: Fine tuned vehicle detection model trained on aerial vehicle OBB dataset from Kaggle.
- **YOLO11m-OBB**: Oriented bounding box segmentation model for precise vehicle boundaries.
- **Training Dataset**: https://www.kaggle.com/datasets/redzapdos123/aerial-vehicle-obb-dataset
- **Vehicle Classes**: Supports detection of cars, trucks, buses, motorcycles, and other vehicle types.

### Tracking Algorithm:
- **Custom BoTSORT**: Multi object tracking with Kalman filter for motion prediction and ReID integration.
- **Feature Extractor**: VeriWild R50 IBN ReID model through newFastReID for appearance based reidentification.
- **ReID Model**: VeriWild R50 IBN model for enhanced vehicle reidentification using newFastReID framework.
- **Motion Model**: Kalman filtering with constant velocity assumption for robust tracking.

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
filterpy>=1.4.5
matplotlib>=3.3.0
tqdm>=4.60.0
```

### Tracking and ReID Libraries:
```
git+https://github.com/WhiteMetagross/newFastReID.git  # Custom FastReID fork for vehicle reidentification
```

## Model Files:

### Required Model Files:
- **YOLO11m Detection Model**: Fine tuned model weights (user trained on aerial vehicle dataset).
- **YOLO11m-OBB Segmentation**: `yolo11m-obb.pt` - Oriented bounding box model.
- **VeriWild ReID Model**: `veriwild_bot_R50-ibn.pth` - R50 IBN ReID weights.
- **ReID Config**: `veriwild_r50_ibn_config.yml` - Configuration for ReID model.

## Installation:

1. **Clone the repository**:
```bash
git clone <repository-url>
cd VehicleSegmentPathTrackerV2
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download model weights**:
   - Place your fine tuned YOLO11m detection model at the specified path.
   - Download YOLO11m-OBB model: `yolo11m-obb.pt`.
   - Obtain VeriWild R50 IBN ReID model: `veriwild_bot_R50-ibn.pth`.
   - Configure ReID model with: `veriwild_r50_ibn_config.yml`.

## Usage:

### Basic Usage:
```bash
python VehicleSegmentPathTrackerv2.py --input input_video.mp4 --output output_video.mp4
```

### Advanced Usage with Custom Models:
```bash
python VehicleSegmentPathTrackerv2.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --yolo-model path/to/custom_yolo_model.pt \
    --segmentation-model path/to/yolo11m-obb.pt \
    --reid-model path/to/veriwild_bot_R50-ibn.pth \
    --reid-config path/to/veriwild_r50_ibn_config.yml \
    --conf-threshold 0.4 \
    --device cuda
```

### Command Line Arguments:
- `--input`: Input video file path (required)
- `--output`: Output video file path (required)
- `--yolo-model`: Path to fine tuned YOLO11m detection model (optional)
- `--segmentation-model`: Path to YOLO11m-OBB segmentation model (optional)
- `--reid-model`: Path to VeriWild R50 IBN ReID model (optional)
- `--reid-config`: Path to ReID model configuration file (optional)
- `--conf-threshold`: Detection confidence threshold (default: 0.35)
- `--device`: Processing device - 'cuda' or 'cpu' (default: 'cuda')

### Programmatic Usage:
```python
from VehicleSegmentPathTrackerv2 import SegmentedVehiclePathTracker

#Initialize tracker:
tracker = SegmentedVehiclePathTracker(
    yolo_model_path="path/to/yolo_model.pt",
    segmentation_model_path="path/to/yolo11m-obb.pt",
    reid_model_path="path/to/veriwild_bot_R50-ibn.pth",
    reid_config_path="path/to/veriwild_r50_ibn_config.yml",
    conf_threshold=0.35,
    device='cuda'
)

#Process video:
success = tracker.track_video("input.mp4", "output.mp4")
```

## Algorithm Details:

### Stacked Detection and Segmentation:
1. **Primary Detection**: Fine tuned YOLO11m identifies vehicle bounding boxes in aerial footage.
2. **Region of Interest (ROI) Extraction**: Expands detection boxes with adaptive padding.
3. **Segmentation**: YOLO11m-OBB processes Region of Interest (ROI) for precise oriented masks.
4. **Mask Refinement**: Morphological operations improve mask quality.
5. **Overlap Scoring**: Combines confidence and spatial overlap for best mask selection.

### Custom BoTSORT Tracking:
1. **Motion Prediction**: Kalman filter with constant velocity model for track state estimation.
2. **Appearance Features**: VeriWild R50 IBN ReID features extracted through newFastReID framework.
3. **Data Association**: Hungarian algorithm for optimal detection-to-track assignment.
4. **Track Management**: Handles track initialization, confirmation, and deletion based on hit streaks.
5. **ReID Integration**: Appearance-based matching for handling occlusions and re-entries.

### Adaptive Thresholding:
The system dynamically adjusts detection parameters based on region size:
- **Small regions** (<10k pixels): Lower thresholds for better sensitivity
- **Medium regions** (10k-50k pixels): Balanced thresholds
- **Large regions** (>50k pixels): Higher thresholds to reduce false positives

### Track Management:
- **Kalman Filtering**: Predicts vehicle motion between frames using constant velocity model
- **Appearance Modeling**: VeriWild R50 IBN features through newFastReID for robust reidentification
- **Track Lifecycle**: Handles track initialization, confirmation, maintenance, and deletion
- **ReID Matching**: Appearance-based association for handling occlusions and re-entries
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

## Troubleshooting:

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

- **YOLO11m**: Ultralytics for object detection and oriented bounding box segmentation models.
- **Custom BoTSORT**: Multi object tracking algorithm with ReID integration.
- **newFastReID**: Enhanced vehicle reidentification capabilities (https://github.com/WhiteMetagross/newFastReID).
- **VeriWild Dataset**: R50 IBN ReID model for vehicle reidentification.
- **Aerial Vehicle Dataset**: Training dataset from Kaggle (https://www.kaggle.com/datasets/redzapdos123/aerial-vehicle-obb-dataset).

## Citation:

If you use this work in your research, please cite:
```bibtex
@software{vehicle_segment_path_tracker,
  title={Vehicle Segment Path Tracker V2},
  author={Your Name},
  year={2025},
  url={https://github.com/WhiteMetagross/ProjectIAV/tree/main/VehicleSegmentPathTrackerv2}
}

```
