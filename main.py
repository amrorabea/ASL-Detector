import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# =================Constants====================
cap = cv2.VideoCapture(0)
noHandsTimer = 0
maxNoHandsTimer = 60*5
# ==============================================
message = ": "
predicted_character = ''
printed = False
# ==============================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# ==============================================

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:

    axy = []
    x_total = []
    y_total = []

    ret, img = cap.read()

    cv2.putText(img, f"Message{message}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                cv2.LINE_AA)

    height, width, _ = img.shape

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        noHandsTimer = 0
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_total.append(x)
                y_total.append(y)

            for i in range(len(hand_landmarks.landmark)):
                axy.append(hand_landmarks.landmark[i].x - min(x_total))
                axy.append(hand_landmarks.landmark[i].y - min(y_total))

        x1 = int(min(x_total) * width) - 10
        y1 = int(min(y_total) * height) - 10

        x2 = int(max(x_total) * width) - 10
        y2 = int(max(y_total) * height) - 10

        if len(axy) == 42:
            printed = False
            prediction = model.predict([np.asarray(axy)])

            predicted_character = prediction[0]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        else:
            if not printed and predicted_character == 'del':
                message = message[:-1]
            elif not printed and predicted_character == 'space':
                message += ' '
            elif not printed:
                message += predicted_character
            printed = True
    else:
        noHandsTimer += 1
        printed = False
    if noHandsTimer >= maxNoHandsTimer:
        break

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break

print(f"Message{message}")
with open('./result.txt', 'w') as f:
    f.write(message[2:])
cap.release()
cv2.destroyAllWindows()
