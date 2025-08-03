import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class EyeTrackerLite:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        self.FACIAL_AXIS_POINTS = {
            'nose_tip': 1,
            'nose_bridge': 6,
            'nose_bottom': 2,
            'forehead_center': 9,
            'forehead_top': 10,
            'chin_tip': 175,
            'chin_bottom': 18,
            'left_temple': 21,
            'right_temple': 251,
            'left_cheek_center': 116,
            'right_cheek_center': 345,
            'left_jaw': 172,
            'right_jaw': 397,
            'left_eyebrow_inner': 70,
            'right_eyebrow_inner': 300,
            'left_eyebrow_outer': 46,
            'right_eyebrow_outer': 276,
            'left_eye_center': 468,
            'right_eye_center': 473,
            'mouth_center': 13,
            'upper_lip': 12,
            'lower_lip': 15
        }
        
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
        
        self.gaze_history = deque(maxlen=10)  # Increased for more smoothing
        self.iris_history = deque(maxlen=3)
        self.pupil_history = deque(maxlen=3)
        self.face_axis_history = deque(maxlen=3)
        
        self.calibrated = False
        self.center_point = None
        self.calibration_samples = []
        self.calibration_threshold = 8
        
        self.show_landmarks = True
        self.aimbot_mode = True
        
        self.pose_weight = 0.15
        self.pose_history = deque(maxlen=3)
        
        self.tracking_confidence = 0.0
        self.eye_openness_threshold = 0.06
        
        self.prev_left_pupil = None
        self.prev_right_pupil = None
        self.fallback_iris_coords = None

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
        
        x_min = max(0, min(xs) - 35)
        x_max = min(w, max(xs) + 35)
        y_min = max(0, min(ys) - 25)
        y_max = min(h, max(ys) + 25)
        
        if x_max <= x_min or y_max <= y_min:
            return None, None
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        return eye_region, (x_min, y_min, x_max - x_min, y_max - y_min)

    def get_anatomical_pupil_center(self, frame, landmarks, iris_indices, eye_indices, eye_bounds):
        if not eye_bounds:
            return None
            
        h, w = frame.shape[:2]
        eye_x, eye_y, eye_w, eye_h = eye_bounds
        
        iris_points = []
        for idx in iris_indices:
            landmark = landmarks.landmark[idx]
            x = int((landmark.x * w - eye_x))
            y = int((landmark.y * h - eye_y))
            if 0 <= x < eye_w and 0 <= y < eye_h:
                iris_points.append((x, y))
        
        if len(iris_points) >= 3:
            center_x = sum(p[0] for p in iris_points) / len(iris_points)
            center_y = sum(p[1] for p in iris_points) / len(iris_points)
            iris_center = (center_x / eye_w, center_y / eye_h)
            self.fallback_iris_coords = iris_center
        
        eye_region = frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
        if eye_region.size == 0:
            if self.fallback_iris_coords:
                return (np.clip(self.fallback_iris_coords[0], 0, 1), np.clip(self.fallback_iris_coords[1], 0, 1))
            return None

        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6)).apply(eye_gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(eye_w, eye_h) * 0.25),
            param1=40,
            param2=25,
            minRadius=int(min(eye_w, eye_h) * 0.08),
            maxRadius=int(min(eye_w, eye_h) * 0.45)
        )
        
        pupil_center = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            if len(circles[0]) > 0:
                circle = circles[0][0]
                pupil_center = (circle[0] / eye_w, circle[1] / eye_h)
        
        if len(iris_points) >= 3:
            iris_center = (center_x / eye_w, center_y / eye_h)
            
            if pupil_center:
                weight = 0.55
                final_x = weight * pupil_center[0] + (1 - weight) * iris_center[0]
                final_y = weight * pupil_center[1] + (1 - weight) * iris_center[1]
                return (np.clip(final_x, 0, 1), np.clip(final_y, 0, 1))
            else:
                return (np.clip(iris_center[0], 0, 1), np.clip(iris_center[1], 0, 1))
        
        if pupil_center:
            return (np.clip(pupil_center[0], 0, 1), np.clip(pupil_center[1], 0, 1))
        
        if self.fallback_iris_coords:
            return (np.clip(self.fallback_iris_coords[0], 0, 1), np.clip(self.fallback_iris_coords[1], 0, 1))
        
        return None

    def get_enhanced_iris_position(self, frame, landmarks, iris_indices, eye_bounds):
        return self.get_anatomical_pupil_center(frame, landmarks, iris_indices, None, eye_bounds)

    def get_eye_openness(self, landmarks, eye_indices):
        if len(eye_indices) < 6:
            return 0.5
            
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            eye_points.append((landmark.x, landmark.y))
        
        eye_array = np.array(eye_points)
        
        top_y = np.min(eye_array[:, 1])
        bottom_y = np.max(eye_array[:, 1])
        left_x = np.min(eye_array[:, 0])
        right_x = np.max(eye_array[:, 0])
        
        vertical_dist = bottom_y - top_y
        horizontal_dist = right_x - left_x
        
        if horizontal_dist > 0:
            openness = vertical_dist / horizontal_dist
            return max(0.08, openness)
        
        return 0.15

    def get_facial_axis_direction(self, landmarks):
        axis_points = {}
        for name, idx in self.FACIAL_AXIS_POINTS.items():
            landmark = landmarks.landmark[idx]
            axis_points[name] = (landmark.x, landmark.y)
        
        nose_tip = axis_points['nose_tip']
        nose_bridge = axis_points['nose_bridge']
        forehead_center = axis_points['forehead_center']
        chin_tip = axis_points['chin_tip']
        left_temple = axis_points['left_temple']
        right_temple = axis_points['right_temple']
        left_cheek = axis_points['left_cheek_center']
        right_cheek = axis_points['right_cheek_center']
        left_eye = axis_points['left_eye_center']
        right_eye = axis_points['right_eye_center']
        mouth_center = axis_points['mouth_center']
        
        face_width = abs(right_temple[0] - left_temple[0])
        face_height = abs(chin_tip[1] - forehead_center[1])
        
        if face_width == 0 or face_height == 0:
            return 0, 0
        
        face_center_x = (left_temple[0] + right_temple[0]) / 2
        face_center_y = (forehead_center[1] + chin_tip[1]) / 2
        
        horizontal_shift = (nose_tip[0] - face_center_x) / face_width * 6.5
        
        vertical_components = []
        
        nose_vertical = (nose_tip[1] - nose_bridge[1]) / face_height * 12.0
        vertical_components.append(nose_vertical)
        
        face_vertical = (nose_tip[1] - face_center_y) / face_height * 8.5
        vertical_components.append(face_vertical)
        
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        eye_to_center = (eye_center_y - face_center_y) / face_height * 7.2
        vertical_components.append(eye_to_center)
        
        mouth_to_center = (mouth_center[1] - face_center_y) / face_height * 5.8
        vertical_components.append(mouth_to_center)
        
        forehead_shift = (forehead_center[1] - face_center_y) / face_height * 4.5
        vertical_components.append(forehead_shift)
        
        chin_shift = (chin_tip[1] - face_center_y) / face_height * 3.2
        vertical_components.append(chin_shift)
        
        total_vertical = sum(vertical_components)
        
        return horizontal_shift, total_vertical

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
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            
            ear_center_x = (left_ear.x + right_ear.x) / 2
            ear_center_y = (left_ear.y + right_ear.y) / 2
            
            if shoulder_width > 0:
                body_offset_x = (nose.x - shoulder_center_x) / shoulder_width * 2.2
                head_offset_x = (nose.x - ear_center_x) / shoulder_width * 2.8
                
                neck_length = abs(ear_center_y - shoulder_center_y)
                if neck_length > 0:
                    head_tilt_y = (nose.y - ear_center_y) / neck_length * 6.5
                    body_tilt_y = (ear_center_y - shoulder_center_y) / neck_length * 3.8
                    total_vertical = head_tilt_y + body_tilt_y
                else:
                    total_vertical = (nose.y - ear_center_y) * 5.5
                
                return head_offset_x + body_offset_x, total_vertical
        except:
            pass
        
        return 0, 0

    def combine_face_and_pose(self, face_h, face_v, pose_h, pose_v):
        face_axis_data = (face_h, face_v, pose_h, pose_v)
        self.face_axis_history.append(face_axis_data)
        
        if len(self.face_axis_history) >= 2:
            avg_face_h = sum(data[0] for data in self.face_axis_history) / len(self.face_axis_history)
            avg_face_v = sum(data[1] for data in self.face_axis_history) / len(self.face_axis_history)
            avg_pose_h = sum(data[2] for data in self.face_axis_history) / len(self.face_axis_history)
            avg_pose_v = sum(data[3] for data in self.face_axis_history) / len(self.face_axis_history)
        else:
            avg_face_h, avg_face_v, avg_pose_h, avg_pose_v = face_h, face_v, pose_h, pose_v
        
        combined_h = avg_face_h * (1 - self.pose_weight) + avg_pose_h * self.pose_weight
        combined_v = avg_face_v * (1 - self.pose_weight) + avg_pose_v * self.pose_weight
        
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
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        
        for name, idx in self.FACIAL_AXIS_POINTS.items():
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
        
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
                    cv2.circle(frame, (x, y), 4, (255, 165, 0), -1)
                except:
                    pass
            
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
        
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        if not face_results.multi_face_landmarks:
            self._show_frame(frame, "No face detected")
            return None
        
        landmarks = face_results.multi_face_landmarks[0]
        pose_landmarks = pose_results.pose_landmarks if pose_results.pose_landmarks else None
        
        if self.show_landmarks:
            self.draw_enhanced_landmarks(frame, landmarks, pose_landmarks)
        
        left_openness = self.get_eye_openness(landmarks, self.LEFT_EYE)
        right_openness = self.get_eye_openness(landmarks, self.RIGHT_EYE)
        
        if left_openness < self.eye_openness_threshold or right_openness < self.eye_openness_threshold:
            if self.prev_left_pupil and self.prev_right_pupil:
                left_pupil, right_pupil = self.prev_left_pupil, self.prev_right_pupil
            else:
                self._show_frame(frame, "Eyes too closed")
                return None
        else:
            left_eye, left_bounds = self.extract_eye_region(frame, landmarks, self.LEFT_EYE)
            right_eye, right_bounds = self.extract_eye_region(frame, landmarks, self.RIGHT_EYE)
            
            if not left_bounds or not right_bounds:
                if self.prev_left_pupil and self.prev_right_pupil:
                    left_pupil, right_pupil = self.prev_left_pupil, self.prev_right_pupil
                else:
                    self._show_frame(frame, "Eye detection failed")
                    return None
            else:
                left_pupil = self.get_anatomical_pupil_center(frame, landmarks, self.LEFT_IRIS, self.LEFT_EYE, left_bounds)
                right_pupil = self.get_anatomical_pupil_center(frame, landmarks, self.RIGHT_IRIS, self.RIGHT_EYE, right_bounds)
                
                if not left_pupil or not right_pupil:
                    if self.prev_left_pupil and self.prev_right_pupil:
                        left_pupil, right_pupil = self.prev_left_pupil, self.prev_right_pupil
                    else:
                        self._show_frame(frame, "Pupil detection failed")
                        return None
                else:
                    self.prev_left_pupil = left_pupil
                    self.prev_right_pupil = right_pupil
        
        self.pupil_history.append((left_pupil, right_pupil))
        
        if len(self.pupil_history) >= 2:
            weights = [0.4, 0.6] if len(self.pupil_history) == 2 else [0.2, 0.3, 0.5]
            weights = weights[-len(self.pupil_history):]
            
            avg_left_x = sum(p[0][0] * w for p, w in zip(self.pupil_history, weights))
            avg_left_y = sum(p[0][1] * w for p, w in zip(self.pupil_history, weights))
            avg_right_x = sum(p[1][0] * w for p, w in zip(self.pupil_history, weights))
            avg_right_y = sum(p[1][1] * w for p, w in zip(self.pupil_history, weights))
            
            left_pupil = (avg_left_x, avg_left_y)
            right_pupil = (avg_right_x, avg_right_y)
        
        face_h_offset, face_v_offset = self.get_facial_axis_direction(landmarks)
        
        if pose_landmarks:
            pose_h_offset, pose_v_offset = self.get_pose_direction(pose_landmarks)
            combined_h_offset, combined_v_offset = self.combine_face_and_pose(
                face_h_offset, face_v_offset, pose_h_offset, pose_v_offset
            )
        else:
            combined_h_offset, combined_v_offset = face_h_offset, face_v_offset
        
        eye_gaze_x = (left_pupil[0] + right_pupil[0]) / 2
        eye_gaze_y = (left_pupil[1] + right_pupil[1]) / 2
        
        # Reduced sensitivity for stability
        if self.aimbot_mode:
            sensitivity_x = 1.7
            sensitivity_y = 2.3
            combined_x = eye_gaze_x + (combined_h_offset * 0.28)
            combined_y = eye_gaze_y + (combined_v_offset * 0.35)
        else:
            sensitivity_x = 1.4
            sensitivity_y = 2.0
            combined_x = eye_gaze_x + (combined_h_offset * 0.22)
            combined_y = eye_gaze_y + (combined_v_offset * 0.28)
        
        combined_x = max(0, min(1, combined_x))
        combined_y = max(0, min(1, combined_y))
        
        if not self.calibrated:
            return self._handle_calibration(frame, combined_x, combined_y, pose_landmarks)
        
        if self.center_point:
            adjusted_x = 0.5 + (combined_x - self.center_point[0]) * sensitivity_x
            adjusted_y = 0.5 + (combined_y - self.center_point[1]) * sensitivity_y
            combined_x = max(0, min(1, adjusted_x))
            combined_y = max(0, min(1, adjusted_y))
        
        coords = (combined_x, combined_y)
        coords = self._smooth_gaze(coords)
        
        if len(self.gaze_history) >= 3:
            recent_gazes = list(self.gaze_history)[-3:]
            variance = np.var(recent_gazes, axis=0)
            self.tracking_confidence = max(0, 1 - (variance[0] + variance[1]) * 10)
        
        if 'left_bounds' in locals() and 'right_bounds' in locals() and left_bounds and right_bounds:
            self._show_tracking(frame, coords, left_bounds, right_bounds, combined_h_offset, combined_v_offset, pose_landmarks)
        else:
            self._show_tracking(frame, coords, None, None, combined_h_offset, combined_v_offset, pose_landmarks)
        
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
        
        mode_text = " FACIAL AXIS TRACK" if self.aimbot_mode else "NORMAL MODE"
        cv2.putText(display_frame, mode_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
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
        
        if len(self.gaze_history) < 2:
            return coords
        
        # More smoothing, favoring recent but averaging more points
        n = len(self.gaze_history)
        weights = np.linspace(0.2, 1.0, n)
        weights = weights / weights.sum()
        
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
                cv2.circle(display_frame, (gaze_x, gaze_y), 10, (0, 255, 255), -1)
                cv2.circle(display_frame, (gaze_x, gaze_y), 13, (255, 255, 255), 2)
        
        mode_color = (255, 0, 255) if self.aimbot_mode else (0, 255, 255)
        mode_text = "FACIAL AXIS" if self.aimbot_mode else "NORMAL"
        cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        pose_status = "POSE+FACE" if pose_landmarks else "FACE ONLY"
        pose_color = (0, 255, 0) if pose_landmarks else (255, 165, 0)
        cv2.putText(display_frame, pose_status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 2)
        
        cv2.putText(display_frame, f"Confidence: {self.tracking_confidence:.2f}", (10, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"H:{face_h:.2f} V:{face_v:.2f}", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Gaze: ({coords[0]:.2f}, {coords[1]:.2f})" if coords else "No gaze", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "A=Mode | L=Landmarks | Q=Quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Eye Tracker', display_frame)

    def _show_frame(self, frame, message):
        display_frame = cv2.flip(frame, 1)
        cv2.putText(display_frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Eye Tracker', display_frame)

    def should_quit(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        elif key == ord('a'):
            self.aimbot_mode = not self.aimbot_mode
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        return False

    def cleanup(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()