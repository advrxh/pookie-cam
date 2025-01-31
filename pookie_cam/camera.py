import time
import base64

import cv2
from upload import Cache

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

bow_image = cv2.imread("bow.png", -1)
bow_image = cv2.resize(
    bow_image, (int(bow_image.shape[1] / 15), int(bow_image.shape[0] / 15))
)


def overlay_bow(frame, x, y, w, h):
    bow_h, bow_w = bow_image.shape[0], bow_image.shape[1]

    bow_x = x + w - bow_w - int(w * 0.05)
    bow_y = y - int(h * 0.1)

    bow_x = max(0, min(bow_x, frame.shape[1] - bow_w))
    bow_y = max(0, min(bow_y, frame.shape[0] - bow_h))

    alpha = bow_image[:, :, 3] / 255.0
    for c in range(3):
        frame[bow_y : bow_y + bow_h, bow_x : bow_x + bow_w, c] = (1 - alpha) * frame[
            bow_y : bow_y + bow_h, bow_x : bow_x + bow_w, c
        ] + alpha * bow_image[:, :, c]


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


CENTER = (frame_width // 2, frame_height // 2)

smile_detected = False
countdown_start = 0
last_capture_time = 0
COUNTDOWN_DURATION = 3
COOLDOWN_DURATION = 10

PINK = (180, 105, 255)
CENTER = (
    (frame_width // 2, frame_height // 2) if "frame_width" in locals() else (320, 240)
)

cache = Cache()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.namedWindow("Pookie Cam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Pookie Cam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    screen_width = int(cv2.getWindowProperty("Pookie Cam", cv2.WND_PROP_FULLSCREEN))
    screen_height = int(cv2.getWindowProperty("Pookie Cam", cv2.WND_PROP_FULLSCREEN))

    cv2.moveWindow(
        "Pookie Cam",
        int((screen_width / 2) - (frame_width / 2)),
        int((screen_height / 2) - (frame_height / 2)),
    )

    current_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), maxSize=(300, 300)
    )

    for x, y, w, h in faces:
        overlay_bow(frame, x, y, w, h)
        roi_gray = gray[y : y + h, x : x + w]

        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=20,
            minSize=(25, 25),
            maxSize=(90, 90),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(smiles) > 0 and (current_time - last_capture_time) > COOLDOWN_DURATION:
            if not smile_detected:
                smile_detected = True
                countdown_start = current_time

    if smile_detected:
        elapsed = current_time - countdown_start
        if elapsed < COUNTDOWN_DURATION:
            cv2.circle(frame, CENTER, 60, PINK, 4)

            countdown_num = 3 - int(elapsed)
            text = str(countdown_num)
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6
            )

            text_x = CENTER[0] - text_width // 2
            text_y = CENTER[1] + text_height // 2
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                PINK,
                6,
                cv2.LINE_AA,
            )
        else:
            _, buffer = cv2.imencode(".png", frame)
            base64_str = base64.b64encode(buffer).decode("utf-8")
            cv2.imwrite(f"./pookie_caps/capture_{int(current_time)}.jpg", frame)
            cache.push(base64_str)
            print("Photo captured!")
            smile_detected = False
            last_capture_time = current_time

    if (current_time - last_capture_time) < COOLDOWN_DURATION:
        cooldown_remaining = COOLDOWN_DURATION - int(current_time - last_capture_time)
        cv2.putText(
            frame,
            f"Cooldown: {cooldown_remaining}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            PINK,
            2,
        )

    cv2.imshow("Pookie cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
