import os
import mediapipe as mp
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

model_dict = pickle.load(open('./model_4.p', 'rb'))
model = model_dict['model_4']
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True,
                       min_detection_confidence=0.4, max_num_hands=2)

# DATA_DIR = '.\Indian'


def exrtractData(result) -> list:
    dataLeft = []
    dataRight = []
    totalData = []
    for handType, handLms in zip(result.multi_handedness, result.multi_hand_landmarks):
        if handType.classification[0].label == 'Left':
            # print('inLeft')
            for i in range(len(handLms.landmark)):
                x = handLms.landmark[i].x
                y = handLms.landmark[i].y
                dataLeft.append(x)
                dataLeft.append(y)

        else:
            # print("inRight")
            for i in range(len(handLms.landmark)):
                x = handLms.landmark[i].x
                y = handLms.landmark[i].y
                dataRight.append(x)
                dataRight.append(y)

    if len(dataLeft) == 0 and len(dataRight) == 42:
        # i.e empty toh zeros sei bhar do
        # 21x 21 y
        # print('inNoLeft')
        dataLeft = [0]*42
    if len(dataRight) == 0 and len(dataLeft) == 42:
        # print('inNoRight')
        dataRight = [0]*42
    totalData.extend(dataLeft)
    totalData.extend(dataRight)
    return totalData


def draw(img, result):
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read feed
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        draw(frame, result)
        frameData = exrtractData(result)

        pred = model.predict([np.asarray(frameData)])

        cv2.putText(frame, pred[0], (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Show to screen
    cv2.imshow('OpenCV Feed', frame)
    referenceImg = cv2.imread('col.png')
    cv2.imshow('Reference', referenceImg)

    # Break gracefully
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
