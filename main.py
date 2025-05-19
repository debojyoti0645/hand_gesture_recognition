import cv2
import mediapipe as mp
import pyautogui
from screeninfo import get_monitors
import math
import win32gui
import win32con
import os
from datetime import datetime
import subprocess
import numpy as np
import time
import webbrowser

screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_hand_type(hand_landmarks, handedness):
    """
    Determine if the detected hand is left or right
    Returns: "Left" or "Right"
    """
    return handedness.classification[0].label

cap = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


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

def is_only_index_up(hand_landmarks):
    """Check if only index finger is up and other fingers (except thumb) are down"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    return (index_tip.y < index_pip.y and  # Index up
            middle_tip.y > middle_pip.y and  # Middle down
            ring_tip.y > ring_pip.y and      # Ring down
            pinky_tip.y > pinky_pip.y)       # Pinky down

def is_index_and_middle_up(hand_landmarks):
    """Check if index and middle fingers are up and other fingers are down"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    return (index_tip.y < index_pip.y and    # Index up
            middle_tip.y < middle_pip.y and   # Middle up
            ring_tip.y > ring_pip.y and       # Ring down
            pinky_tip.y > pinky_pip.y)        # Pinky down

def is_thumb_up_or_down(hand_landmarks):
    """Check if thumb is pointing up or down, returns: 1 for up, -1 for down, 0 for neither"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    vertical_direction = thumb_tip.y - thumb_mcp.y
    
    is_extended = calculate_distance(thumb_tip, thumb_mcp) > calculate_distance(thumb_ip, thumb_mcp)
    
    if not is_extended:
        return 0
    
    if vertical_direction < -0.1:  # Pointing up
        return 1
    elif vertical_direction > 0.1:  # Pointing down
        return -1
    return 0

def is_palm_open(hand_landmarks):
    """Check if palm is open (all fingers extended)"""
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]
    
    for tip_id, pip_id in fingers:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        if tip.y > pip.y:  # If any finger is not extended
            return False
    return True

def is_all_fingers_down(hand_landmarks):
    """Check if all fingers (except thumb) are pointing down"""
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]
    
    for tip_id, pip_id in fingers:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        if tip.y < pip.y:  # If any finger is pointing up
            return False
    return True

application_opened = False
last_left_click_time = 0
last_right_click_time = 0
last_double_click_time = 0
last_triple_click_time = 0
last_scroll_time = 0  

CLICK_COOLDOWN = 0.5  
DOUBLE_CLICK_COOLDOWN = 1.0  
TRIPLE_CLICK_COOLDOWN = 1.5  
SCROLL_COOLDOWN = 0.5  
YOUTUBE_COOLDOWN = 2.0  

last_youtube_time = 0
youtube_opened = False  

def draw_ui_overlay(frame, gesture_status):
    sidebar_width = 480
    
    cv2.putText(frame, "Gesture Control Panel", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (10, 40), (sidebar_width-10, 40), (255, 255, 255), 1)

    y_offset = 80  
    for hand in ["Right Hand", "Left Hand"]:
        cv2.putText(frame, f"{hand}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30  
        
        hand_active = any(gesture_status[hand].values())
        hand_status_color = (0, 255, 0) if hand_active else (0, 0, 255)
        cv2.circle(frame, (20, y_offset-15), 5, hand_status_color, -1)
        
        for gesture, status in gesture_status[hand].items():
            if status:
                cv2.rectangle(frame, (10, y_offset-20), (sidebar_width-10, y_offset+5),
                             (0, 153, 0), -1)
            else:
                cv2.rectangle(frame, (10, y_offset-20), (sidebar_width-10, y_offset+5),
                             (64, 64, 64), -1)
            
            cv2.putText(frame, gesture, (50, y_offset-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_offset += 25  
        
        y_offset += 35 

    instructions = [
        "Instructions:",
        "Right Hand:",
        "- Thumb + Index: Left Click",
        "- Thumb + Middle: Right Click",
        "- Index Up: Move Mouse",
        "- Fist: Take Screenshot",
        "",
        "Left Hand:",
        "- Index Up: Double Click",
        "- Index + Middle Up: Triple Click",
        "- Thumb Up: Scroll Up",
        "- Thumb Down: Scroll Down",
        "",
        "Both Hands:",
        "- All Fingers Down: Open YouTube",
        "",
        "Press 'Q' to quit"
    ]
    
    y_offset = 400 
    cv2.line(frame, (10, y_offset-10), (sidebar_width-10, y_offset-10), 
             (255, 255, 255), 1)
    
    y_offset += 20  
    for instruction in instructions:
        if instruction == "":  
            y_offset += 15
            continue
            
        cv2.putText(frame, instruction, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 20  

    return frame

screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(screenshots_dir, exist_ok=True)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        gesture_status = {
            "Right Hand": {
                "Left Click": False,
                "Right Click": False,
                "Mouse Move": False,
                "Fist": False,
                "All Fingers Down": False  # Add this line
            },
            "Left Hand": {
                "Double Click": False,
                "Triple Click": False,
                "Scroll Up": False,
                "Scroll Down": False,
                "All Fingers Down": False  # Add this line
            }
        }

        if result.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)):
                
                hand_type = get_hand_type(hand_landmarks, handedness)
                
                color = (0, 255, 0) if hand_type == "Right" else (255, 0, 0)
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

                if hand_type == "Right":
                    hand_key = "Right Hand"
                else:
                    hand_key = "Left Hand"

                if hand_type == "Right":
                    current_time = time.time()
                    
                    if is_left_click_gesture(hand_landmarks):
                        gesture_status[hand_key]["Left Click"] = True
                        if current_time - last_left_click_time > CLICK_COOLDOWN:
                            pyautogui.click(button='left')
                            last_left_click_time = current_time
                    
                    elif is_right_click_gesture(hand_landmarks):
                        gesture_status[hand_key]["Right Click"] = True
                        if current_time - last_right_click_time > CLICK_COOLDOWN:
                            pyautogui.click(button='right')
                            last_right_click_time = current_time
                    
                    elif is_index_finger_only_up_and_thumb_down(hand_landmarks):
                        gesture_status[hand_key]["Mouse Move"] = True
                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        cursor_x = int(index_finger_tip.x * screen_width)
                        cursor_y = int(index_finger_tip.y * screen_height)
                        pyautogui.moveTo(cursor_x, cursor_y)
                    elif is_fist(hand_landmarks) and not application_opened:
                        gesture_status[hand_key]["Fist"] = True
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = os.path.join(screenshots_dir, f"screenshot_{timestamp}.png")
                        screenshot = pyautogui.screenshot()
                        screenshot.save(screenshot_path)
                        application_opened = True
                        cv2.waitKey(2000)
                        application_opened = False

                else:
                    current_time = time.time()
                    
                    if is_only_index_up(hand_landmarks):
                        gesture_status[hand_key]["Double Click"] = True
                        if current_time - last_double_click_time > DOUBLE_CLICK_COOLDOWN:
                            pyautogui.doubleClick()
                            last_double_click_time = current_time
                    elif is_index_and_middle_up(hand_landmarks):
                        gesture_status[hand_key]["Triple Click"] = True
                        if current_time - last_triple_click_time > TRIPLE_CLICK_COOLDOWN:
                            pyautogui.click(clicks=3) 
                            last_triple_click_time = current_time
                    elif not is_only_index_up(hand_landmarks) and not is_index_and_middle_up(hand_landmarks):
                        thumb_direction = is_thumb_up_or_down(hand_landmarks)
                        current_time = time.time()
                        
                        if thumb_direction == 1: 
                            gesture_status[hand_key]["Scroll Up"] = True
                            if current_time - last_scroll_time > SCROLL_COOLDOWN:
                                pyautogui.scroll(120) 
                                last_scroll_time = current_time
                        elif thumb_direction == -1:
                            gesture_status[hand_key]["Scroll Down"] = True
                            if current_time - last_scroll_time > SCROLL_COOLDOWN:
                                pyautogui.scroll(-120) 
                                last_scroll_time = current_time

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            left_hand = None
            right_hand = None
            
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                if get_hand_type(hand_landmarks, handedness) == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks
            
            if left_hand and right_hand:
                current_time = time.time()
                if (is_all_fingers_down(left_hand) and is_all_fingers_down(right_hand) and
                    not youtube_opened and current_time - last_youtube_time > YOUTUBE_COOLDOWN):
                    webbrowser.open('https://www.youtube.com')
                    youtube_opened = True
                    last_youtube_time = current_time
                    gesture_status["Left Hand"]["All Fingers Down"] = True
                    gesture_status["Right Hand"]["All Fingers Down"] = True
        
        if youtube_opened and time.time() - last_youtube_time > YOUTUBE_COOLDOWN:
            youtube_opened = False

        frame = np.zeros((720, 500, 3), dtype=np.uint8) 
        
        frame = draw_ui_overlay(frame, gesture_status)
        
        cv2.imshow("Gesture Control Status", frame)
        
        hwnd = win32gui.FindWindow(None, "Gesture Control Status")
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 
                                screen_width - 500, 0,  
                                500, 720,
                                win32con.SWP_SHOWWINDOW)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
