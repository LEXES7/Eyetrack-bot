import pyautogui
import time
import numpy as np

class GameController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        self.last_x = self.screen_w // 2
        self.last_y = self.screen_h // 2
        self.smoothing = 0.3
        
        self.last_move_time = time.time()
        self.min_move_interval = 0.001
        
        self.dead_zone_x = 0.008
        self.dead_zone_y = 0.008
        
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_damping = 0.9
        
        self.aimbot_mode = False

    def set_aimbot_mode(self, enabled):
        self.aimbot_mode = enabled
        if enabled:
            self.smoothing = 0.4
            self.dead_zone_x = 0.005
            self.dead_zone_y = 0.005
            self.min_move_interval = 0.001
        else:
            self.smoothing = 0.3
            self.dead_zone_x = 0.008
            self.dead_zone_y = 0.008
            self.min_move_interval = 0.001

    def move_mouse(self, gaze_coords):
        if not gaze_coords:
            return
        
        current_time = time.time()
        if current_time - self.last_move_time < self.min_move_interval:
            return
        
        gaze_x, gaze_y = gaze_coords
        
        center_x, center_y = 0.5, 0.5
        
        if (abs(gaze_x - center_x) < self.dead_zone_x and 
            abs(gaze_y - center_y) < self.dead_zone_y):
            return
        
        target_x = (1 - gaze_x) * self.screen_w
        target_y = gaze_y * self.screen_h
        
        diff_x = target_x - self.last_x
        diff_y = target_y - self.last_y
        
        movement_speed = np.sqrt(diff_x**2 + diff_y**2)
        
        if self.aimbot_mode:
            if movement_speed > 150:
                smoothing_factor = 0.5
            else:
                smoothing_factor = 0.4
        else:
            if movement_speed > 200:
                smoothing_factor = 0.4
            else:
                smoothing_factor = 0.3
        
        smooth_x = self.last_x + (diff_x * smoothing_factor)
        smooth_y = self.last_y + (diff_y * smoothing_factor)
        
        smooth_x = max(0, min(self.screen_w, smooth_x))
        smooth_y = max(0, min(self.screen_h, smooth_y))
        
        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y))
            self.last_x = smooth_x
            self.last_y = smooth_y
            self.last_move_time = current_time
        except:
            pass

    def click(self):
        try:
            pyautogui.click()
        except:
            pass