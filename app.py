#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import math
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import pyautogui
import keyboard
import time

def resetZoom():
    pyautogui.hotkey('ctrl', '0')

def zoom_in():
    keyboard.press('ctrl')
    pyautogui.scroll(100)
    keyboard.release('ctrl')

def zoom_out():
    keyboard.press('ctrl')
    pyautogui.scroll(-100)
    keyboard.release('ctrl')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)  # Reduced width
    parser.add_argument("--height", type=int, default=360)  # Reduced height
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    keypoint_classifier = KeyPointClassifier()    
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    prev_distance = None
    prev_index_y = None
    mode, frame_skip_counter = 0, 0

    while cap.isOpened():
        frame_skip_counter += 1
        if frame_skip_counter % 2 == 0:
            continue  # Skip every other frame to improve speed

        fps = cvFpsCalc.get()
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = image.copy()
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                h, w, _ = image.shape
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                if keypoint_classifier_labels[hand_sign_id] == "Open":
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if prev_distance is None:
                        prev_distance = distance

                    if abs(distance - prev_distance) > 4:
                        if distance > prev_distance:
                            zoom_in()
                        else:
                            zoom_out()
                    prev_distance = distance
                elif keypoint_classifier_labels[hand_sign_id] == "Close":
                    resetZoom()
                elif keypoint_classifier_labels[hand_sign_id] == "Pointer":
                    if prev_index_y is None:
                        prev_index_y = y2
                    else:
                        if prev_index_y > args.height/2:
                            pyautogui.scroll(int(prev_index_y - (args.height/2)))
                            prev_index_y = y2
                        
                        if prev_index_y < args.height/2:
                            pyautogui.scroll(int(prev_index_y - (args.height/2)))
                            prev_index_y = y2
                        
                        prev_index_y = y2

        cv.imshow('Hand Gesture Recognition', debug_image)
        if cv.waitKey(10) & 0xFF == 27:  # ESC key to break
            break

    cap.release()
    cv.destroyAllWindows()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
