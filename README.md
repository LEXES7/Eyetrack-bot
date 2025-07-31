# Eyetrack-bot

Eyetrack bot is an advanced eye tracking system that uses real time computer vision to control mouse movement through gaze detection. Built with MediaPipe and OpenCV, it provides precise cursor control for accessibility and productivity applications.

## Features

- **Real-time Eye Tracking**: Uses MediaPipe face mesh for accurate iris detection
- **Precision Mode**: High-accuracy targeting mode for detailed work
- **Face Landmark Visualization**: Complete face tracking with numbered reference points
- **Stability Control**: Filters micro-movements for smooth cursor control
- **Calibration System**: Easy setup with visual feedback
- **Cross-platform**: Works on macOS, Windows, and Linux


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LEXES7/Eyetrack-bot.git
   cd Eyetrack-bot
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

## Usage

1. **Run the application:**
   ```bash
   cd src
   python main.py
   ```

2. **Calibration:**
   - Look at the green center circle
   - Press **SPACE** 5 times while looking at center
   - Wait for "Calibrated" message

3. **Controls:**
   - **SPACE**: Calibrate (during setup)
   - **A**: Toggle Precision/Normal mode
   - **L**: Toggle face landmarks display
   - **Q**: Quit application

## Modes

### Normal Mode
- Standard sensitivity for general use
- Smooth cursor movement
- Good for browsing and regular tasks

### Precision Mode
- Higher sensitivity and accuracy
- Reduced smoothing for faster response
- Enhanced targeting display
- Optimized for detailed work

## Technical Details

- **Face Detection**: MediaPipe Face Mesh (468 landmarks)
- **Iris Tracking**: 4-point iris detection per eye
- **Stability**: 3-frame consistency requirement
- **Smoothing**: Weighted average with velocity compensation
- **Performance**: ~17-25 FPS depending on hardware

## System Requirements

- **Camera**: Any USB webcam or built-in camera
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Modern multi-core processor recommended

## Troubleshooting

**Low FPS:**
- Close other camera applications
- Reduce video resolution in code
- Disable face landmarks (press L)

**Cursor not moving:**
- Complete calibration process
- Check camera permissions
- Ensure good lighting conditions

**Inaccurate tracking:**
- Recalibrate the system
- Improve lighting setup
- Sit closer to camera

## Use Cases

- **Accessibility**: Hands-free computer control for users with mobility limitations
- **Productivity**: Quick navigation and cursor positioning
- **Research**: Eye tracking studies and human-computer interaction
- **Education**: Learning computer vision and eye tracking technologies

## Disclaimer

This software is intended for educational, research, and accessibility purposes. Users are responsible for complying with applicable laws and terms of service when using this software.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

Created by Sachintha