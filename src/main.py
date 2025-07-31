from eye_tracker import EyeTrackerLite
from game_controller import GameController
import time

def main():
    print("Starting Enhanced Eye Tracker with Pose Detection...")
    
    eye_tracker = EyeTrackerLite()
    game_controller = GameController()
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    
    print("Eye tracker initialized successfully!")
    print("Enhanced with shoulder and neck tracking for better accuracy")
    print("Controls:")
    print("  SPACE - Calibrate")
    print("  A - Toggle Eyetrack/Normal mode")
    print("  L - Toggle landmarks display")
    print("  Q - Quit")
    
    try:
        while True:
            gaze_coords = eye_tracker.get_gaze()
            
            if gaze_coords and eye_tracker.calibrated:
                game_controller.set_aimbot_mode(eye_tracker.aimbot_mode)
                game_controller.move_mouse(gaze_coords)
            
            frame_count += 1
            current_time = time.time()
            
            if current_time - last_fps_time > 3.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                mode = "eyetracking" if eye_tracker.aimbot_mode else "NORMAL"
                print(f"FPS: {fps:.1f} | Mode: {mode} | Enhanced Pose Tracking")
                last_fps_time = current_time
            
            if eye_tracker.should_quit():
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        eye_tracker.cleanup()
        print("Eye tracker closed successfully!")

if __name__ == "__main__":
    main()