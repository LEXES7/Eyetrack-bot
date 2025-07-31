# EyeTrack-Auto

EyeTrack-Auto is a project designed to automate gameplay in sniper minigames using eye tracking technology. The application utilizes OpenCV and MediaPipe for eye detection and tracking, allowing users to control game actions through their gaze.

## Project Structure

```
EyeTrack-Auto
├── src
│   ├── main.py                # Entry point of the application
│   ├── eye_tracker.py         # Eye tracking functionality
│   ├── game_controller.py     # Game interaction management
│   ├── calibration.py          # Calibration of eye tracking system
│   └── utils
│       ├── __init__.py        # Initializes the utils module
│       ├── image_processing.py # Image processing utilities
│       └── coordinates.py      # Coordinate conversion functions
├── models
│   └── eye_tracking_model.py   # Eye tracking model definitions
├── config
│   ├── settings.py            # Configuration settings
│   └── game_configs.json      # Game-specific settings
├── tests
│   ├── __init__.py            # Initializes the tests module
│   ├── test_eye_tracker.py     # Unit tests for EyeTracker
│   └── test_game_controller.py # Unit tests for GameController
├── requirements.txt            # Project dependencies
├── setup.py                    # Project packaging
├── .gitignore                  # Files to ignore in version control
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/EyeTrack-Auto.git
   cd EyeTrack-Auto
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python src/main.py
   ```

2. Follow the on-screen instructions to calibrate the eye tracking system.

3. Enjoy automated gameplay in sniper minigames!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.# Eyetrack-bot
