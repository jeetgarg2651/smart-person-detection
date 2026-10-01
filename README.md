# smart-person-detection
# Smart Person Detection

A Python computer-vision project for detecting and tracking people in a video using Ultralytics YOLO and OpenCV.

## Features

- Person detection and tracking with persistent track IDs
- Restricted-area selection with the mouse
- Alerts when a tracked person leaves the selected area
- Person count and total unique-person count
- Audio alert using Windows `winsound`

## Requirements

- Windows
- Python 3.9 or newer
- A working camera/video display environment

Install the Python dependencies:

```powershell
pip install ultralytics opencv-python
```

## Files

- `main.py` - restricted-area person tracking application
- `trackeres/track_test.py` - simple YOLOv8n tracking test
- `input.mp4` - input video used by the applications
- `yolov8n.pt` - YOLOv8 nano model

`main.py` currently loads `yolov8m.pt`. Download that model separately if it is not available locally, or change the model path in `main.py` to `yolov8n.pt`.

## Run

From the project directory:

```powershell
python main.py
```

Select the restricted area in the first window. Press `q` to stop the video.

To run the lightweight tracking test:

```powershell
python trackeres/track_test.py
```

## Notes

The YOLO weight files are large binary files. Keep only the model files needed for your selected workflow, and use a compatible Ultralytics model when downloading replacements.# smart-person-detection
