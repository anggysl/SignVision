import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

def detect_gesture():
    cap = cv2.VideoCapture(0)

    result_text = "-"

    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm = handLms.landmark

            # RULE sederhana
            if lm[8].y > lm[6].y:
                result_text = "A"
            else:
                result_text = "B"

    cap.release()
    return result_text