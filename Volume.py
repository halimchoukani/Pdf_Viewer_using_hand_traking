import cv2
import mediapipe as mp
import math
import pyautogui
import keyboard
import time

def zoom_in():
    # Press and hold Ctrl while scrolling up
    keyboard.press('ctrl')
    pyautogui.scroll(100)  # Scroll up to zoom in
    keyboard.release('ctrl')

def zoom_out():
    # Press and hold Ctrl while scrolling down
    keyboard.press('ctrl')
    pyautogui.scroll(-100)  # Scroll down to zoom out
    keyboard.release('ctrl')


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Capture video feed
cap = cv2.VideoCapture(0)

# Variable to store previous distance
prev_distance = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip and convert the image color
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get coordinates of thumb tip (landmark[4]) and index finger tip (landmark[8])
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Convert relative coordinates to pixel coordinates
            height, width, _ = image.shape
            x1, y1 = int(thumb_tip.x * width), int(thumb_tip.y * height)
            x2, y2 = int(index_tip.x * width), int(index_tip.y * height)

            # Draw a line between thumb tip and index finger tip
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate the Euclidean distance between thumb and index finger
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # If this is the first frame, set the initial distance
            if prev_distance is None:
                prev_distance = distance

            # Compare with the previous distance and zoom in or out
            if distance > prev_distance and distance-prev_distance>4:  # Zoom in
                zoom_in()
            elif distance  < prev_distance and prev_distance-distance>4:  # Zoom out
                zoom_out()

            # Update the previous distance
            prev_distance = distance

        
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
