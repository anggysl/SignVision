import cv2
import os

label = "terimakasih"

folder = f"dataset/{label}"

os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Kamera gagal dibuka")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(frame,
                f"Collecting {label}: {count}",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    # tekan s
    if key == ord('s'):

        filename = f"{folder}/{count}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Berhasil simpan: {filename}")

        count += 1

    # tekan q keluar
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()