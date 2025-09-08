# VehiclePathBoTSORTTrackerV2

A vehicle tracking system that integrates YOLO detection, FastReID appearance features, and BoTSORT tracking for robust multi object vehicle tracking.

## Overview:

This project is an **independent consumer** of the FastReID library, implementing a complete vehicle tracking pipeline that combines:

- **YOLO Detection**: Real time vehicle detection using YOLOv8
- **FastReID Integration**: Uses the VeRiWild pre-trained model for vehicle re-identification
- **BoTSORT Tracking**: Advanced tracking algorithm with motion prediction and appearance matching

## Architecture

```
Input Video → YOLO Detection → Vehicle Crops → FastReID Features → BoTSORT Tracking → Output Tracks
```

## Dependencies

This project **uses** the FastReID library as an external dependency:

1. **FastReID Library**: Located at `..\newFastReid\`
2. **VeRiWild Model**: `..\veriwild_bot_R50-ibn.pth`
3. **Configuration**: `..\veriwild_r50_ibn_config.yml`

## Installation

### 1. Install newFastReID Library:

First, install the newFastReID library:

```bash
git clone https://github.com/WhiteMetagross/newFastReID.git
cd newFastReID
pip install -e .
```

### 2. Install VehiclePathBoTSORTTracker Dependencies

```bash
cd VehiclePathBoTSORTTracker
pip install -r requirements.txt
```

## Usage:

### Basic Webcam Tracking:

```bash
python vehiclePathTrackerv3.py --source webcam
```

### Command Line Options

```bash
python vehiclePathTrackerv3.py --help
```

Options:
- `--source`: Input source (webcam, video)
- `--input`: Input video file (for video source)
- `--output`: Output video file
- `--yolo-model`: YOLO model path (default: yolov8n.pt)
- `--conf-threshold`: Detection confidence threshold (default: 0.5)
- `--device`: Device (cuda/cpu)

### Programmatic Usage:

```python
from vehiclePathTrackerv3 import VehiclePathTracker

#Initialize tracker.
tracker = VehiclePathTracker(
    yolo_model_path="yolo11m.pt",
    reid_model_path=r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
    reid_config_path=r"C:\Users\Xeron\OneDrive\Documents\FastReid\veriwild_r50_ibn_config.yml",
    device='cuda'
)

#Process frame.
tracks, detections, processing_time = tracker.process_frame(frame)

#Draw results.
annotated_frame = tracker.draw_results(frame, tracks, detections)
```

## Project Structure:

```
VehiclePathBoTSORTTrackerV2/
├── vehiclePathTrackerv3.py    #Main tracking implementation.
├── botsort.py                 #BoTSORT algorithm with FastReID integration.
├── test_integration.py        #Integration tests.
├── requirements.txt           #Dependencies.
└── README.md                  #This file.
```

## Key Components

### 1. `vehiclePathTrackerv3.py`
- Main vehicle tracking system.
- YOLO detection integration.
- Complete tracking pipeline.
- Command line interface.

### 2. `botsort.py`
- BoTSORT tracking algorithm.
- FastReID feature extraction.
- Kalman filter for motion prediction.
- Data association with Hungarian algorithm.

## FastReID Integration:

This project integrates with the FastReID library through:

```python
#Import FastReID library:
import fastreid
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

#Use VeRiWild model for vehicle ReID:
class FastReIDFeatureExtractor:
    def __init__(self, model_path, config_path, device='cuda'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.model = build_model(self.cfg)
        #Load VeRiWild weights...
```

## Troubleshooting:

### Common Issues:

**1. FastReID Import Error:**
```
ImportError: No module named 'fastreid'
```
**Solution:**: Install FastReID library:
```bash
git clone https://github.com/WhiteMetagross/newFastReID.git
cd newFastReid
pip install -e .
```

**2. Model File Not Found:**
```
FileNotFoundError: VeRiWild model not found
```
**Solution**: Verify model path:
```bash
ls "..\veriwild_bot_R50-ibn.pth"
```

**3. CUDA Out of Memory:**
```
RuntimeError: CUDA out of memory
```
**Solution:** Use CPU or reduce batch size:
```bash
python vehiclePathTrackerv3.py --device cpu
```

All tests passed. The tracking system is ready to use.
```

## Key Features:

- **Independent Project**: Uses FastReID as external library dependency.
- **VeRiWild Model**: Pre-trained vehicle re-identification model.
- **Real time Performance**: Optimized for live video processing
- **Robust Tracking**: Reduces identity switches with appearance features
- **Easy Integration**: Simple API and command line interface

This project successfully demonstrates how to use the FastReID library as a dependency for building advanced vehicle tracking systems.
