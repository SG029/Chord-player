import cv2
import mediapipe as mp
import pygame
import os
import time
import numpy as np

# Initialize pygame mixer
pygame.mixer.init()

# Set up file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHORDS_DIR = os.path.join(BASE_DIR, "Chords")

# Check if Chords directory exists
if not os.path.exists(CHORDS_DIR):
    print(f"Error: Folder 'Chords' not found. Please create it and add:")
    print("- D.mp3 (Left Index)")
    print("- Bm.mp3 (Left Thumb)")
    print("- G.mp3 (Right Index)")
    print("- A.mp3 (Right Thumb)")
    exit()

# Sound file mapping (now using thumb instead of middle)
SOUND_MAPPING = {
    ("left", "index"): "D.mp3",
    ("left", "thumb"): "Bm.mp3", 
    ("right", "index"): "G.mp3",
    ("right", "thumb"): "A.mp3"
}

# Verify all sound files exist
missing_files = [f for f in SOUND_MAPPING.values() if not os.path.exists(os.path.join(CHORDS_DIR, f))]
if missing_files:
    print(f"Missing files in 'Chords': {', '.join(missing_files)}")
    exit()

# Load sounds
sounds = {}
for (hand, finger), filename in SOUND_MAPPING.items():
    sounds[(hand, finger)] = pygame.mixer.Sound(os.path.join(CHORDS_DIR, filename))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

# Finger tracking
finger_states = {key: False for key in sounds}
last_played = {key: 0 for key in sounds}
COOLDOWN = 0.5  # Seconds between plays

def get_finger_box(landmarks, finger_tip, finger_dip, finger_pip, finger_mcp, image_shape):
    """Get bounding box around a finger"""
    h, w = image_shape[:2]
    
    # Get coordinates of key points
    tip = [int(landmarks.landmark[finger_tip].x * w), int(landmarks.landmark[finger_tip].y * h)]
    dip = [int(landmarks.landmark[finger_dip].x * w), int(landmarks.landmark[finger_dip].y * h)]
    pip = [int(landmarks.landmark[finger_pip].x * w), int(landmarks.landmark[finger_pip].y * h)]
    mcp = [int(landmarks.landmark[finger_mcp].x * w), int(landmarks.landmark[finger_mcp].y * h)]
    
    # Create bounding box
    x_coords = [tip[0], dip[0], pip[0], mcp[0]]
    y_coords = [tip[1], dip[1], pip[1], mcp[1]]
    
    padding = 20
    x_min = max(0, min(x_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_min = max(0, min(y_coords) - padding)
    y_max = min(h, max(y_coords) + padding)
    
    return (x_min, y_min, x_max, y_max)

def is_finger_up(landmarks, finger_tip, finger_pip, is_thumb=False):
    """Check if finger is extended"""
    tip = landmarks.landmark[finger_tip]
    pip = landmarks.landmark[finger_pip]
    
    if is_thumb:
        # Special handling for thumb - different direction
        return tip.x < pip.x if hand_type == "left" else tip.x > pip.x
    else:
        # For other fingers, check y-coordinate
        mcp = landmarks.landmark[finger_pip - 2]
        dip = landmarks.landmark[finger_pip - 1]
        return (tip.y < pip.y and tip.y < dip.y and tip.y < mcp.y)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    current_time = time.time()
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            hand_type = handedness.classification[0].label.lower()
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(250, 44, 90), thickness=2, circle_radius=2))
            
            # Check fingers (now including thumb)
            fingers_to_check = {
                "index": (
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_DIP,
                    mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    False
                ),
                "thumb": (
                    mp_hands.HandLandmark.THUMB_TIP,
                    mp_hands.HandLandmark.THUMB_IP,
                    mp_hands.HandLandmark.THUMB_MCP,
                    mp_hands.HandLandmark.WRIST,  # Using wrist as reference for thumb
                    True
                )
            }
            
            for finger, (tip, dip, pip, mcp, is_thumb) in fingers_to_check.items():
                key = (hand_type, finger)
                if key in sounds:
                    # Get finger bounding box
                    box = get_finger_box(hand_landmarks, tip, dip, pip, mcp, frame.shape)
                    x_min, y_min, x_max, y_max = box
                    
                    # Check if finger is up (special handling for thumb)
                    finger_up = is_finger_up(hand_landmarks, tip, pip, is_thumb)
                    
                    # Draw bounding box with color based on state
                    color = (0, 255, 0) if finger_up else (0, 0, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Label the box
                    cv2.putText(frame, f"{hand_type} {finger}", (x_min, y_min-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Handle sound playback
                    if finger_up:
                        if not finger_states[key] and (current_time - last_played[key] > COOLDOWN):
                            sounds[key].play()
                            last_played[key] = current_time
                        finger_states[key] = True
                    else:
                        finger_states[key] = False

    # Display instructions
    y = 30
    cv2.putText(frame, "Finger Chord Player", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    
    for (hand, finger), filename in SOUND_MAPPING.items():
        cv2.putText(frame, f"{hand.title()} {finger}: {filename.replace('.mp3','')}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25
    
    cv2.imshow('Finger Chord Player', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()