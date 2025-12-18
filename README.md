# Hand Gesture Mouse Control

A real-time hand tracking application that lets you control your computer mouse using hand gestures. Move your cursor with your index finger and click by pinching your thumb and index finger together.

## Features

### Hand-tracking
- Real-time hand detection and 21-point landmark tracking
- Support for tracking up to 2 hands simultaneously
- Visual skeletal overlay on detected hands
- Smooth, low-latency performance

### Gesture controls
- **Cursor Movement**: Your index finger position controls the mouse cursor
- **Click Action**: Pinch your thumb and index finger together to click
- **Smart Smoothing**: Prevents jittery cursor movements
- **Mirror Mode**: Natural mirrored display for intuitive interaction

### Feedback
- Live hand skeleton visualization
- On-screen pinch detection indicator
- Console logging for click events

## Requirements
- Python 3.8 or higher
- Webcam
- macOS, Linux, or Windows

## Installation

1. clone the repo:
   ```bash
   git clone <your-repo-url>
   cd hand-tracking
   ```

2. create a venv (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   and if you're on Windows:
   ```bash
   venv\Scripts\activate
   ```

3. install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. download the hand-tracking model:
   ```bash
   curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

## Usage

run the app:
```bash
python main.py
```

### Controls
- move your index finger to control the cursor
- pinch your thumb and index finger together to click
- Press 'q' to quit the application

## How it works

### Tech stack
- MediaPipe: Google's ML framework for hand landmark detection
- OpenCV: real-time video processing & display
- PyAutoGUI: system-level mouse control (i've only tested till IDE)

### Pipeline
1. captures the video frames from your webcam
2. converts frames to an rgb format so mediapipe can use it
3. detects hands & extracts the pre-labeled 21 landmark points per hand
4. maps index finger position to screen coordinates
5. calculates the distance b/w thumb & index finger tips
6. triggers click when pinch distance falls below a certain threshold

### Key parameters

You can adjust these in `main.py`:

```python
# cursor smoothing (0.0-0.9, least smoothing to highest)
smoothing = 0.5

# pinch detection threshold (lower = more sensitive)
pinch_threshold = 0.05

# hand detection confidence
min_hand_detection_confidence = 0.5
min_tracking_confidence = 0.5
```

## Dependencies

```
mediapipe==0.10.31     # hand tracking ML model
opencv-python          # vid capture + display
pyautogui              # mouse control
```

## Troubleshooting (i went through ts too dw)

### if your camera isn't detected
- make sure your webcam is connected and not being used by another application
- on macOS, you may need to grant camera permissions in system preferences

### cursor is jittery for some reason
- increase the `smoothing` value (try 0.7 or 0.8)
- make sure you have good lighting
- keep your hand at a steady distance from the camera

### clicks are too sensitive or not sensitive enough
- adjust the `pinch_threshold` value:
  increase (ex: 0.07) for less sensitivity OR decrease (ex: 0.03) for more sensitivity

### "No module named 'mediapipe.framework'"
- this means you're using an incompatible MediaPipe version
- make sure you have MediaPipe 0.10.30 or higher installed
- run `pip install --upgrade mediapipe`

### model file not found
- make sure `hand_landmarker.task` is in the same directory as `main.py`
- re-download the model using the curl command from step 4 in [installation](#installation)

## license

MIT License - feel free to use this project for personal or commercial purposes.

## ack

- MediaPipe by Google for the hand tracking model
- OpenCV community for video processing tools
- PyAutoGUI devs for mouse control capabilities
