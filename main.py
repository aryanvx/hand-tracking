import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen_width, screen_height = pyautogui.size()

def calculate_distance(point1, point2):
    """calculate Euclidean distance b/w 2 landmarks"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def draw_landmarks_on_image(image, detection_result):
    """draw hand landmarks on the image"""
    hand_landmarks_list = detection_result.hand_landmarks
    h, w, _ = image.shape

    for hand_landmarks in hand_landmarks_list:

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),            # index
            (0, 9), (9, 10), (10, 11), (11, 12),       # middle
            (0, 13), (13, 14), (14, 15), (15, 16),     # ring
            (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
            (5, 9), (9, 13), (13, 17)                  # palm
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
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.HandLandmarker.create_from_options(options)

    cam = cv2.VideoCapture(0)
    timestamp = 0

    smooth_x, smooth_y = 0, 0
    smoothing = 0.5

    was_pinching = False

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("empty frame")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = detector.detect_for_video(mp_image, timestamp)
        timestamp += 1

        if detection_result.hand_landmarks:

            hand_landmarks = detection_result.hand_landmarks[0]

            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            cursor_x = screen_width - (index_tip.x * screen_width)
            cursor_y = index_tip.y * screen_height

            smooth_x = smooth_x * smoothing + cursor_x * (1 - smoothing)
            smooth_y = smooth_y * smoothing + cursor_y * (1 - smoothing)

            pyautogui.moveTo(smooth_x, smooth_y)

            pinch_distance = calculate_distance(thumb_tip, index_tip)

            pinch_threshold = 0.05
            is_pinching = pinch_distance < pinch_threshold

            if is_pinching and not was_pinching:
                pyautogui.click()
                print("clicked")

            was_pinching = is_pinching

            if is_pinching:
                cv2.putText(frame, "pinch detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        annotated_frame = draw_landmarks_on_image(frame, detection_result)

        cv2.imshow("hand tracking", cv2.flip(annotated_frame, 1))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_tracking_on_webcam()