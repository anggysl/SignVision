import keras
import numpy as np

from flask import Flask, render_template, Response, jsonify, send_file

import cv2
import mediapipe as mp

from gtts import gTTS

app = Flask(__name__)

# =========================
# LOAD MODEL AI
# =========================
model = keras.models.load_model("model.h5", compile=False)
# LABEL DATASET
labels = ["halo", "saya", "anggy", "terimakasih"]

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =========================
# GLOBAL VARIABLE
# =========================
latest_gesture = "-"

# =========================
# PREDICT AI
# =========================
def predict_gesture(frame):

    # Resize gambar
    img = cv2.resize(frame, (64, 64))

    # Normalize
    img = img / 255.0

    # Tambah dimensi
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Ambil index terbesar
    class_index = np.argmax(prediction)

    # Return label
    return labels[class_index]

# =========================
# GENERATE CAMERA FRAME
# =========================
def gen_frames():

    global latest_gesture

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Flip camera
        frame = cv2.flip(frame, 1)

        # RGB mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        gesture_text = "-"

        # Jika tangan terdeteksi
        if result.multi_hand_landmarks:

            for handLms in result.multi_hand_landmarks:

                # Draw landmark
                mp_draw.draw_landmarks(
                    frame,
                    handLms,
                    mp_hands.HAND_CONNECTIONS
                )

                # Predict AI
                gesture_text = predict_gesture(frame)

        latest_gesture = gesture_text

        # Tampilkan text
        cv2.putText(
            frame,
            f"{gesture_text}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2
        )

        # Convert frame
        _, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# =========================
# ROUTE HOME
# =========================
@app.route('/')
def index():
    return render_template('index.html')

# =========================
# ROUTE VIDEO
# =========================
@app.route('/video')
def video():

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# ROUTE GESTURE
# =========================
@app.route('/gesture')
def gesture():

    return jsonify({
        "result": latest_gesture
    })

# =========================
# ROUTE SPEAK
# =========================
@app.route('/speak/<text>')
def speak(text):

    tts = gTTS(
        text=text,
        lang='id'
    )

    tts.save("voice.mp3")

    return send_file(
        "voice.mp3",
        mimetype="audio/mpeg"
    )

# =========================
# RUN APP
# =========================
if __name__ == '__main__':

    app.run(debug=True)