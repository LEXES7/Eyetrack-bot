import cv2
import mediapipe as mp
import numpy as np
import time

class EyeTrackerLite:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        self.FACE_CONNECTIONS = [
            (10, 151), (151, 9), (9, 10),
            (234, 127), (127, 162), (162, 21), (21, 54),
            (454, 356), (356, 389), (389, 251), (251, 284),
            (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
            (17, 314), (314, 405), (405, 320), (320, 375), (375, 291)
        ]
        
        self.EYE_CONNECTIONS = [
            (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
            (362, 382), (382, 381), (381, 380), (380, 374), (374, 373)
        ]
        
        self.gaze_history = []
        self.history_size = 3
        
        self.calibrated = False
        self.center_point = None
        self.calibration_samples = []
        self.calibration_threshold = 5
        
        self.show_landmarks = True
        self.aimbot_mode = True
        
        # Pose integration settings
        self.pose_weight = 0.25
        self.pose_history = []
        self.pose_history_size = 3

    def extract_eye_region(self, frame, landmarks, eye_indices):
        h, w = frame.shape[:2]
        
        points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x_min = max(0, min(xs) - 15)
        x_max = min(w, max(xs) + 15)
        y_min = max(0, min(ys) - 15)
        y_max = min(h, max(ys) + 15)
        
        if x_max <= x_min or y_max <= y_min:
            return None, None
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        return eye_region, (x_min, y_min, x_max - x_min, y_max - y_min)

    def get_iris_position(self, frame, landmarks, iris_indices, eye_bounds):
        if not eye_bounds:
            return None
        
        h, w = frame.shape[:2]
        iris_points = []
        
        for idx in iris_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            iris_points.append((x, y))
        
        if not iris_points:
            return None
        
        center_x = sum(p[0] for p in iris_points) / len(iris_points)
        center_y = sum(p[1] for p in iris_points) / len(iris_points)
        
        eye_x, eye_y, eye_w, eye_h = eye_bounds
        
        relative_x = (center_x - eye_x) / eye_w if eye_w > 0 else 0.5
        relative_y = (center_y - eye_y) / eye_h if eye_h > 0 else 0.5
        
        return (relative_x, relative_y)

    def get_pose_direction(self, pose_landmarks):
        if not pose_landmarks:
            return 0, 0
        
        try:
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            
            ear_center_x = (left_ear.x + right_ear.x) / 2
            ear_center_y = (left_ear.y + right_ear.y) / 2
            
            if shoulder_width > 0:
                body_offset_x = (nose.x - shoulder_center_x) / shoulder_width * 1.5
                head_offset_x = (nose.x - ear_center_x) / shoulder_width * 2.0
                head_offset_y = (nose.y - ear_center_y) * 3.0
                
                return head_offset_x + body_offset_x, head_offset_y
        except:
            pass
        
        return 0, 0

    def get_enhanced_face_direction(self, landmarks):
        nose_tip = landmarks.landmark[1]
        nose_bridge = landmarks.landmark[168]
        left_cheek = landmarks.landmark[234]
        right_cheek = landmarks.landmark[454]
        forehead = landmarks.landmark[10]
        chin = landmarks.landmark[152]
        left_eye_corner = landmarks.landmark[33]
        right_eye_corner = landmarks.landmark[362]
        
        face_width = abs(right_cheek.x - left_cheek.x)
        face_height = abs(chin.y - forehead.y)
        
        if face_width == 0 or face_height == 0:
            return 0, 0
        
        face_center_x = (left_cheek.x + right_cheek.x) / 2
        face_center_y = (forehead.y + chin.y) / 2
        
        horizontal_offset = (nose_tip.x - face_center_x) / face_width * 4
        
        vertical_nose_offset = (nose_tip.y - nose_bridge.y) / face_height * 6
        vertical_face_offset = (nose_tip.y - face_center_y) / face_height * 3
        vertical_offset = vertical_nose_offset + vertical_face_offset
        
        eye_center_y = (left_eye_corner.y + right_eye_corner.y) / 2
        eye_vertical_offset = (eye_center_y - face_center_y) / face_height * 2
        vertical_offset += eye_vertical_offset
        
        return horizontal_offset, vertical_offset

    def combine_face_and_pose(self, face_h, face_v, pose_h, pose_v):
        # Smooth pose data
        self.pose_history.append((pose_h, pose_v))
        if len(self.pose_history) > self.pose_history_size:
            self.pose_history.pop(0)
        
        if len(self.pose_history) >= 2:
            smooth_pose_h = sum(p[0] for p in self.pose_history) / len(self.pose_history)
            smooth_pose_v = sum(p[1] for p in self.pose_history) / len(self.pose_history)
        else:
            smooth_pose_h, smooth_pose_v = pose_h, pose_v
        
        # Combine face and pose
        combined_h = face_h * (1 - self.pose_weight) + smooth_pose_h * self.pose_weight
        combined_v = face_v * (1 - self.pose_weight) + smooth_pose_v * self.pose_weight
        
        return combined_h, combined_v

    def draw_enhanced_landmarks(self, frame, landmarks, pose_landmarks=None):
        h, w = frame.shape[:2]
        
        for connection in self.FACE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]
            
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            
            cv2.line(frame, start_point, end_point, (0, 255, 255), 1)
        
        for connection in self.EYE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]
            
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        key_points = [(1, "NOSE"), (10, "FOREHEAD"), (152, "CHIN"), (234, "L_CHEEK"), (454, "R_CHEEK")]
        for idx, label in key_points:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{idx}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw pose landmarks if available
        if pose_landmarks:
            pose_points = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_EAR,
                self.mp_pose.PoseLandmark.RIGHT_EAR,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]
            
            for landmark_idx in pose_points:
                try:
                    landmark = pose_landmarks.landmark[landmark_idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (255, 165, 0), -1)
                except:
                    pass
            
            # Draw shoulder line
            try:
                left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                ls_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                rs_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                cv2.line(frame, ls_point, rs_point, (255, 165, 0), 3)
            except:
                pass

    def get_gaze(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process both face and pose
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        if not face_results.multi_face_landmarks:
            self._show_frame(frame, "No face detected")
            return None
        
        landmarks = face_results.multi_face_landmarks[0]
        pose_landmarks = pose_results.pose_landmarks if pose_results.pose_landmarks else None
        
        if self.show_landmarks:
            self.draw_enhanced_landmarks(frame, landmarks, pose_landmarks)
        
        left_eye, left_bounds = self.extract_eye_region(frame, landmarks, self.LEFT_EYE)
        right_eye, right_bounds = self.extract_eye_region(frame, landmarks, self.RIGHT_EYE)
        
        if not left_bounds or not right_bounds:
            self._show_frame(frame, "Eye detection failed")
            return None
        
        left_iris = self.get_iris_position(frame, landmarks, self.LEFT_IRIS, left_bounds)
        right_iris = self.get_iris_position(frame, landmarks, self.RIGHT_IRIS, right_bounds)
        
        if not left_iris or not right_iris:
            self._show_frame(frame, "Iris detection failed")
            return None
        
        face_h_offset, face_v_offset = self.get_enhanced_face_direction(landmarks)
        
        # Get pose direction and combine with face
        if pose_landmarks:
            pose_h_offset, pose_v_offset = self.get_pose_direction(pose_landmarks)
            combined_h_offset, combined_v_offset = self.combine_face_and_pose(
                face_h_offset, face_v_offset, pose_h_offset, pose_v_offset
            )
        else:
            combined_h_offset, combined_v_offset = face_h_offset, face_v_offset
        
        eye_gaze_x = (left_iris[0] + right_iris[0]) / 2
        eye_gaze_y = (left_iris[1] + right_iris[1]) / 2
        
        if self.aimbot_mode:
            combined_x = eye_gaze_x + (combined_h_offset * 0.35)
            combined_y = eye_gaze_y + (combined_v_offset * 0.45)
        else:
            combined_x = eye_gaze_x + (combined_h_offset * 0.2)
            combined_y = eye_gaze_y + (combined_v_offset * 0.3)
        
        combined_x = max(0, min(1, combined_x))
        combined_y = max(0, min(1, combined_y))
        
        if not self.calibrated:
            return self._handle_calibration(frame, combined_x, combined_y, pose_landmarks)
        
        if self.center_point:
            if self.aimbot_mode:
                sensitivity_x = 2.3
                sensitivity_y = 2.1
            else:
                sensitivity_x = 1.8
                sensitivity_y = 1.5
                
            adjusted_x = 0.5 + (combined_x - self.center_point[0]) * sensitivity_x
            adjusted_y = 0.5 + (combined_y - self.center_point[1]) * sensitivity_y
            combined_x = max(0, min(1, adjusted_x))
            combined_y = max(0, min(1, adjusted_y))
        
        coords = (combined_x, combined_y)
        coords = self._smooth_gaze(coords)
        
        self._show_tracking(frame, coords, left_bounds, right_bounds, combined_h_offset, combined_v_offset, pose_landmarks)
        
        return coords

    def _handle_calibration(self, frame, x, y, pose_landmarks=None):
        display_frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        center = (w//2, h//2)
        cv2.circle(display_frame, center, 25, (0, 255, 0), -1)
        cv2.circle(display_frame, center, 30, (255, 255, 255), 3)
        
        progress = len(self.calibration_samples) / self.calibration_threshold
        cv2.rectangle(display_frame, (50, h - 50), (int(50 + progress * 300), h - 30), (0, 255, 0), -1)
        cv2.rectangle(display_frame, (50, h - 50), (350, h - 30), (255, 255, 255), 2)
        
        cv2.putText(display_frame, f"Look at center - Press SPACE ({len(self.calibration_samples)}/{self.calibration_threshold})", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        mode_text = "EYETRACK MODE" if self.aimbot_mode else "NORMAL MODE"
        cv2.putText(display_frame, mode_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        pose_status = "POSE+FACE" if pose_landmarks else "FACE ONLY"
        pose_color = (0, 255, 0) if pose_landmarks else (255, 165, 0)
        cv2.putText(display_frame, pose_status, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self.calibration_samples.append((x, y))
            if len(self.calibration_samples) >= self.calibration_threshold:
                avg_x = sum(s[0] for s in self.calibration_samples) / len(self.calibration_samples)
                avg_y = sum(s[1] for s in self.calibration_samples) / len(self.calibration_samples)
                self.center_point = (avg_x, avg_y)
                self.calibrated = True
        elif key == ord('a'):
            self.aimbot_mode = not self.aimbot_mode
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        
        cv2.imshow('Eye Tracker', display_frame)
        return None

    def _smooth_gaze(self, coords):
        self.gaze_history.append(coords)
        if len(self.gaze_history) > self.history_size:
            self.gaze_history.pop(0)
        
        if len(self.gaze_history) < 2:
            return coords
        
        if self.aimbot_mode:
            weights = [0.3, 0.7][:len(self.gaze_history)]
        else:
            weights = [0.4, 0.6][:len(self.gaze_history)]
        
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        smooth_x = sum(coord[0] * weight for coord, weight in zip(self.gaze_history, weights))
        smooth_y = sum(coord[1] * weight for coord, weight in zip(self.gaze_history, weights))
        
        return (smooth_x, smooth_y)

    def _show_tracking(self, frame, coords, left_bounds, right_bounds, face_h, face_v, pose_landmarks=None):
        display_frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        if left_bounds and right_bounds:
            lx, ly, lw, lh = left_bounds
            rx, ry, rw, rh = right_bounds
            
            cv2.rectangle(display_frame, (w - lx - lw, ly), (w - lx, ly + lh), (0, 255, 0), 2)
            cv2.rectangle(display_frame, (w - rx - rw, ry), (w - rx, ry + rh), (0, 255, 0), 2)
        
        if coords:
            gaze_x = int((1 - coords[0]) * w)
            gaze_y = int(coords[1] * h)
            
            if self.aimbot_mode:
                cv2.circle(display_frame, (gaze_x, gaze_y), 8, (255, 0, 255), -1)
                cv2.circle(display_frame, (gaze_x, gaze_y), 12, (255, 255, 255), 2)
                cv2.line(display_frame, (gaze_x - 15, gaze_y), (gaze_x + 15, gaze_y), (255, 0, 255), 2)
                cv2.line(display_frame, (gaze_x, gaze_y - 15), (gaze_x, gaze_y + 15), (255, 0, 255), 2)
            else:
                cv2.circle(display_frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
                cv2.circle(display_frame, (gaze_x, gaze_y), 13, (255, 255, 255), 2)
        
        mode_color = (255, 0, 255) if self.aimbot_mode else (0, 255, 255)
        mode_text = "EYETRACK" if self.aimbot_mode else "NORMAL"
        cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        pose_status = "POSE+FACE" if pose_landmarks else "FACE ONLY"
        pose_color = (0, 255, 0) if pose_landmarks else (255, 165, 0)
        cv2.putText(display_frame, pose_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        cv2.putText(display_frame, f"H:{face_h:.2f} V:{face_v:.2f}", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Gaze: ({coords[0]:.2f}, {coords[1]:.2f})" if coords else "No gaze", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "A=Aimbot | L=Landmarks | Q=Quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Eye Tracker', display_frame)

    def _show_frame(self, frame, message):
        display_frame = cv2.flip(frame, 1)
        cv2.putText(display_frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Eye Tracker', display_frame)

    def should_quit(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def cleanup(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()