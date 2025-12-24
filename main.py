import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import random
import time
import numpy as np

class Fruit:
    def __init__(self, screen_width, screen_height):
        self.x = random.randint(100, screen_width - 100)
        self.y = screen_height
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-15, -20)
        self.gravity = 0.5
        self.radius = random.randint(30, 50)
        self.sliced = False
        self.slice_time = 0

        fruit_colors = [
            (0, 100, 255),    # orange
            (0, 255, 0),      # green apple
            (0, 0, 255),      # red apple
            (0, 255, 255),    # banana
            (128, 0, 128),    # grape
        ]

        self.color = random.choice(fruit_colors)

    def update(self):
        if not self.sliced:
            self.vy += self.gravity
            self.x += self.vx
            self.y += self.vy

    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)
        else:
            elapsed = time.time() - self.slice_time
            if elapsed < 0.5:
                offset = int(elapsed * 100)
                cv2.ellipse(frame, (int(self.x - offset), int(self.y)), (self.radius, self.radius), 0, 90, 270, self.color, -1)
                cv2.ellipse(frame, (int(self.x + offset), int(self.y)), (self.radius, self.radius), 0, 270, 90, self.color, -1)

                
    def is_off_screen(self, screen_height):
        return self.y > screen_height + 100
    
    def check_slice(self, finger_trail):
        if self.sliced:
            return False
            
        for point in finger_trail:
            if point is not None:
                px, py = point
                distance = math.sqrt((px - self.x)**2 + (py - self.y)**2)
                if distance < self.radius:
                    self.sliced = True
                    self.slice_time = time.time()
                    return True
        return False


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen_width, screen_height = pyautogui.size()

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def draw_landmarks_on_image(image, detection_result):
    if not detection_result.hand_landmarks:
        return image
    
    hand_landmarks_list = detection_result.hand_landmarks
    h, w, _ = image.shape
    
    for hand_landmarks in hand_landmarks_list:

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),             # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),             # index
            (0, 9), (9, 10), (10, 11), (11, 12),        # middle
            (0, 13), (13, 14), (14, 15), (15, 16),      # ring
            (0, 17), (17, 18), (18, 19), (19, 20),      # pinky
            (5, 9), (9, 13), (13, 17)                   # palm
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    
    return image

def run_hand_tracking_on_webcam():
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.HandLandmarker.create_from_options(options)

    fruits = []
    finger_trail = []
    paused = False
    score = 0
    lives = 3
    spawn_timer = 0
    was_pinching = False

    cam = cv2.VideoCapture(0)
    timestamp = 0

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("empty frame")
            continue
        
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect_for_video(mp_image, timestamp)
        timestamp += 1

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]

            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
            finger_trail.append(finger_pos)
            if len(finger_trail) > 20:
                finger_trail.pop(0)

            pinch_distance = calculate_distance(thumb_tip, index_tip)
            pinch_threshold = 0.05
            is_pinching = pinch_distance < pinch_threshold

            if is_pinching and not was_pinching:
                paused = not paused
                print("paused:", paused)

            was_pinching = is_pinching

            if not paused:
                spawn_timer += 1
                if spawn_timer > 60:
                    fruits.append(Fruit(w, h))
                    spawn_timer = 0

        for fruit in fruits[:]:
            if not paused and fruit.check_slice(finger_trail):
                score += 10
                print("fruit sliced! score:", score)

            fruit.update()
            fruit.draw(frame)

            if fruit.is_off_screen(h):
                fruits.remove(fruit)
                if not fruit.sliced and not paused:
                    lives -= 1
                    print("you suck! lives:", lives)

        if lives <= 0:
            paused = True
            cv2.putText(frame, "GAME OVER - Press R", (w // 2 - 250, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Lives: {lives}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        state = "PAUSED" if paused else "PLAY"
        
        cv2.putText(frame, state, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for i in range(1, len(finger_trail)):
            cv2.line(frame, finger_trail[i - 1], finger_trail[i], (0, 0, 255), 3)

        landmark_frame = frame.copy()
        landmark_frame = draw_landmarks_on_image(landmark_frame, detection_result)

        cv2.imshow("hand tracking", landmark_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            fruits.clear()
            score = 0
            lives = 3
            paused = False

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_tracking_on_webcam()