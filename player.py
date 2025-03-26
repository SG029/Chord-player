import cv2
import mediapipe as mp
import pygame
import os
import time

def setup_pygame():
    pygame.mixer.init()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chords_dir = os.path.join(base_dir, "Chords")
    return chords_dir

def verify_files(chords_dir):
    sound_mapping = {
        ("left", "index"): "D.mp3",
        ("left", "thumb"): "Bm.mp3", 
        ("right", "index"): "G.mp3",
        ("right", "thumb"): "A.mp3"
    }
    
    if not os.path.exists(chords_dir):
        print("Error: 'Chords' folder not found. Create it and add:")
        print("- D.mp3, Bm.mp3, G.mp3, A.mp3")
        exit()

    missing = [f for f in sound_mapping.values() if not os.path.exists(os.path.join(chords_dir, f))]
    if missing:
        print(f"Missing files: {', '.join(missing)}")
        exit()

    sounds = {}
    for (hand, finger), filename in sound_mapping.items():
        sounds[(hand, finger)] = pygame.mixer.Sound(os.path.join(chords_dir, filename))
    
    return sounds, sound_mapping

def setup_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    return mp_hands, hands

def get_finger_box(landmarks, tip, dip, pip, mcp, img_shape):
    h, w = img_shape[:2]
    tip_pos = [int(landmarks.landmark[tip].x * w), int(landmarks.landmark[tip].y * h)]
    dip_pos = [int(landmarks.landmark[dip].x * w), int(landmarks.landmark[dip].y * h)]
    pip_pos = [int(landmarks.landmark[pip].x * w), int(landmarks.landmark[pip].y * h)]
    mcp_pos = [int(landmarks.landmark[mcp].x * w), int(landmarks.landmark[mcp].y * h)]
    
    x_coords = [tip_pos[0], dip_pos[0], pip_pos[0], mcp_pos[0]]
    y_coords = [tip_pos[1], dip_pos[1], pip_pos[1], mcp_pos[1]]
    
    padding = 20
    x_min = max(0, min(x_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_min = max(0, min(y_coords) - padding)
    y_max = min(h, max(y_coords) + padding)
    
    return x_min, y_min, x_max, y_max

def check_finger_state(landmarks, tip, pip, is_thumb, hand_type):
    tip_pos = landmarks.landmark[tip]
    pip_pos = landmarks.landmark[pip]
    
    if is_thumb:
        return tip_pos.x < pip_pos.x if hand_type == "left" else tip_pos.x > pip_pos.x
    else:
        mcp = landmarks.landmark[pip - 2]
        dip = landmarks.landmark[pip - 1]
        return (tip_pos.y < pip_pos.y and tip_pos.y < dip.y and tip_pos.y < mcp.y)

def process_hands(frame, results, hands, mp_hands, sounds, sound_mapping, finger_states, last_played):
    h, w = frame.shape[:2]
    current_time = time.time()
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label.lower()
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(250, 44, 90), thickness=2, circle_radius=2))
            
            fingers = {
                "index": (mp_hands.HandLandmark.INDEX_FINGER_TIP,
                         mp_hands.HandLandmark.INDEX_FINGER_DIP,
                         mp_hands.HandLandmark.INDEX_FINGER_PIP,
                         mp_hands.HandLandmark.INDEX_FINGER_MCP,
                         False),
                "thumb": (mp_hands.HandLandmark.THUMB_TIP,
                          mp_hands.HandLandmark.THUMB_IP,
                          mp_hands.HandLandmark.THUMB_MCP,
                          mp_hands.HandLandmark.WRIST,
                          True)
            }
            
            for finger, (tip, dip, pip, mcp, is_thumb) in fingers.items():
                key = (hand_type, finger)
                if key in sounds:
                    x_min, y_min, x_max, y_max = get_finger_box(hand_landmarks, tip, dip, pip, mcp, frame.shape)
                    finger_up = check_finger_state(hand_landmarks, tip, pip, is_thumb, hand_type)
                    
                    color = (0, 255, 0) if finger_up else (0, 0, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, f"{hand_type} {finger}", (x_min, y_min-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if finger_up:
                        if not finger_states[key] and (current_time - last_played[key] > 0.5):
                            sounds[key].play()
                            last_played[key] = current_time
                        finger_states[key] = True
                    else:
                        finger_states[key] = False

def show_instructions(frame, sound_mapping):
    y = 30
    cv2.putText(frame, "Finger Chord Player", (10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    
    for (hand, finger), filename in sound_mapping.items():
        cv2.putText(frame, f"{hand.title()} {finger}: {filename.replace('.mp3','')}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25

def main():
    chords_dir = setup_pygame()
    sounds, sound_mapping = verify_files(chords_dir)
    mp_hands, hands = setup_mediapipe()
    
    finger_states = {key: False for key in sounds}
    last_played = {key: 0 for key in sounds}
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        process_hands(frame, results, hands, mp_hands, sounds, sound_mapping, finger_states, last_played)
        show_instructions(frame, sound_mapping)
        
        cv2.imshow('Finger Chord Player', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()