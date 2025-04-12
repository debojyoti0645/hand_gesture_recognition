import cv2
import mediapipe as mp
import pyautogui
from screeninfo import get_monitors
import math
import win32gui
import win32con
import subprocess
import numpy as np

# Get screen dimensions for scaling
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )

# Gesture detection functions remain unchanged

def is_left_click_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return calculate_distance(thumb_tip, index_tip) < 0.03

def is_right_click_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    return calculate_distance(thumb_tip, middle_tip) < 0.03

def is_index_finger_only_up_and_thumb_down(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    return index_tip.y < index_mcp.y and thumb_tip.y > thumb_ip.y

def is_fist(hand_landmarks):
    for finger_tip in [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]:
        finger = hand_landmarks.landmark[finger_tip]
        base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        if finger.y < base.y:
            return False
    return True

# Flag to prevent multiple application launches
application_opened = False

# New function to draw a UI overlay
def draw_ui_overlay(frame, gesture_status):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (250, frame.shape[0]), (0, 0, 0), -1)  # Sidebar
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Gesture status box
    y_offset = 50
    for gesture, status in gesture_status.items():
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.putText(frame, f"{gesture}: {'Active' if status else 'Inactive'}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 40

# Initialize MediaPipe Hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        result = hands.process(rgb_frame)
        gesture_status = {
            "Left Click": False,
            "Right Click": False,
            "Mouse Move": False,
            "Fist": False
        }

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Left Click
                if is_left_click_gesture(hand_landmarks):
                    gesture_status["Left Click"] = True
                    pyautogui.click(button='left')

                # Right Click
                elif is_right_click_gesture(hand_landmarks):
                    gesture_status["Right Click"] = True
                    pyautogui.click(button='right')

                # Mouse Movement
                elif is_index_finger_only_up_and_thumb_down(hand_landmarks):
                    gesture_status["Mouse Move"] = True
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cursor_x = int(index_finger_tip.x * screen_width)
                    cursor_y = int(index_finger_tip.y * screen_height)
                    pyautogui.moveTo(cursor_x, cursor_y)

                # Fist Gesture
                elif is_fist(hand_landmarks) and not application_opened:
                    gesture_status["Fist"] = True
                    subprocess.Popen(["notepad.exe"])
                    application_opened = True

        # Draw updated UI overlay
        draw_ui_overlay(frame, gesture_status)

        # Display the frame
        cv2.imshow("Gesture Recognition", frame)

        # Ensure window stays on top
        hwnd = win32gui.FindWindow(None, "Gesture Recognition")
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
