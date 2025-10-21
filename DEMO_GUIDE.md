# Reachy Mini Simulation Demo Guide

This guide provides examples for controlling Reachy Mini in MuJoCo simulation.

## Starting the Simulation

### Basic (empty scene)
```bash
mjpython -m reachy_mini.daemon.app.main --sim
```

### With table and objects
```bash
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
```

The daemon will:
- Open a MuJoCo viewer window with the robot
- Start a REST API server on `http://localhost:8000`
- Allow connections via the Python SDK

## Demo Scripts

### 1. Basic Movement Test (`test_simulation.py`)
Simple head movements to test the setup:
- Move head up and roll
- Tilt forward
- Rotate left and right
- Reset to neutral

**Run**: `python test_simulation.py`

### 2. Antenna Expressions (`demo_antennas.py`)
Shows how to use antennas for emotional expression:
- Neutral position
- Happy (antennas outward)
- Curious (one antenna raised)
- Sad (antennas drooping)
- Wiggling animation

**Run**: `python demo_antennas.py`

### 3. Combined Behaviors (`demo_combined.py`)
Expressive behaviors combining head and antennas:
- Looking around curiously
- Happy greeting
- Nodding "yes"
- Shaking "no"
- Thinking pose

**Run**: `python demo_combined.py`

### 4. Camera Feed (`demo_camera.py`)
Access and display the robot's camera feed:
- Opens OpenCV window showing camera view
- Robot looks at table and scans environment
- Press 'q' to close camera window early

**Run**: `python demo_camera.py`

### 5. Choreographed Movements (`demo_choreography.py`)
Complex, smooth movement patterns:
- Greeting sequence
- Wave motion
- Spiral motion
- Figure-8 pattern
- Excited wiggle
- Environmental scan

**Run**: `python demo_choreography.py`

## Python SDK Quick Reference

### Basic Setup
```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy_mini:
    # Your code here
    pass
```

### Head Control
```python
# Create a head pose (all parameters optional)
pose = create_head_pose(
    x=0.0,      # X position in meters (or mm if mm=True)
    y=0.0,      # Y position in meters
    z=0.0,      # Z position in meters
    roll=0.0,   # Roll in radians (or degrees if degrees=True)
    pitch=0.0,  # Pitch in radians
    yaw=0.0,    # Yaw in radians
    degrees=False,  # Set to True to use degrees
    mm=False        # Set to True to use millimeters
)

# Move to target pose
reachy_mini.goto_target(head=pose, duration=2.0)

# Get current head pose
current_pose = reachy_mini.get_current_head_pose()
```

### Antenna Control
```python
# Set antenna positions (radians)
# [left_antenna, right_antenna]
# Negative = antenna tilts left/inward
# Positive = antenna tilts right/outward

reachy_mini.set_target_antenna_joint_positions([0.0, 0.0])  # Neutral
reachy_mini.set_target_antenna_joint_positions([-1.5, 1.5])  # Happy
reachy_mini.set_target_antenna_joint_positions([0.5, -0.5])  # Sad

# Get current antenna positions
positions = reachy_mini.get_present_antenna_joint_positions()
```

### Camera
```python
# Read a frame from the camera
frame = reachy_mini.media.camera.read()  # Returns numpy array (BGR format)

# Display with OpenCV
import cv2
cv2.imshow("Camera", frame)
cv2.waitKey(1)
```

### Joint Positions
```python
# Get all current joint positions
head_joints, antenna_joints = reachy_mini.get_current_joint_positions()
```

### Motor Control
```python
# Enable/disable motors
reachy_mini.enable_motors()
reachy_mini.disable_motors()

# Gravity compensation (for real robot)
reachy_mini.enable_gravity_compensation()
reachy_mini.disable_gravity_compensation()
```

## REST API Quick Reference

### Get Robot State
```bash
# Full state
curl 'http://localhost:8000/api/state/full'

# Head pose only
curl 'http://localhost:8000/api/state/head/pose'

# Antenna positions
curl 'http://localhost:8000/api/state/antennas'
```

### API Documentation
Open `http://localhost:8000/docs` in your browser for interactive API documentation.

## Tips and Tricks

1. **Smooth Movements**: Use longer durations (2.0-3.0 seconds) for natural-looking movements
2. **Expressive Antennas**: Combine antenna movements with head poses for more expression
3. **Camera Lag**: The camera feed might have slight lag in simulation
4. **Coordinate System**:
   - X: forward/backward
   - Y: left/right
   - Z: up/down
   - Yaw: rotation around Z (left/right head turn)
   - Pitch: rotation around Y (up/down head tilt)
   - Roll: rotation around X (head roll/tilt)

## Troubleshooting

### Daemon won't start
- Make sure you're using `mjpython` on macOS
- Check that port 8000 isn't already in use
- Verify MuJoCo is installed: `pip list | grep mujoco`

### Connection errors
- Make sure the daemon is running
- Check that `http://localhost:8000` is accessible
- Default timeout is 5 seconds, increase if needed:
  ```python
  reachy_mini = ReachyMini(timeout=10.0)
  ```

### Camera not working
- Use `reachy_mini.media.camera.read()` not `get_frame()`
- Camera returns BGR format (OpenCV standard)
- Returns None if no frame available

## Next Steps

1. Explore the [official documentation](https://github.com/pollen-robotics/reachy_mini)
2. Check out the [conversation demo](https://github.com/pollen-robotics/reachy_mini_conversation_demo)
3. Try controlling the real robot (if you have hardware)
4. Experiment with the REST API for remote control
5. Build your own AI applications!

## Stopping the Simulation

Press `CTRL+C` in the terminal where the daemon is running, or close the MuJoCo viewer window.
