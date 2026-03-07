"""
Göz Kırpma Sayacı
OpenCV + MediaPipe FaceLandmarker ile gerçek zamanlı göz kırpma tespiti
"""

import cv2
import math
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model URL ve yolu
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def download_model():
    """Model dosyası yoksa indir"""
    if not os.path.exists(MODEL_PATH):
        print("Model indiriliyor...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model indirildi.")


# Göz landmark indeksleri (MediaPipe Face Landmarker - 478 nokta)
# Sol göz: 33, 160, 158, 133, 153, 144
# Sağ göz: 362, 385, 387, 263, 373, 380
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye_indices):
    """Göz en-boy oranı (EAR) hesapla - kırpma tespiti için"""
    # Dikey mesafeler
    p1 = landmarks[eye_indices[1]]
    p2 = landmarks[eye_indices[5]]
    vertical = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # Yatay mesafe
    p3 = landmarks[eye_indices[0]]
    p4 = landmarks[eye_indices[3]]
    horizontal = math.sqrt((p3.x - p4.x) ** 2 + (p3.y - p4.y) ** 2)

    if horizontal == 0:
        return 0
    return vertical / horizontal


def main():
    download_model()

    # FaceLandmarker oluştur (VIDEO modu - webcam için)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    blink_count = 0
    ear_threshold = 0.22
    ear_consecutive_frames = 0
    frames_to_blink = 2
    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # mp.Image oluştur
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Yüz landmark'larını tespit et
        detection_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33  # ~30 FPS için

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2

            if ear < ear_threshold:
                ear_consecutive_frames += 1
            else:
                if ear_consecutive_frames >= frames_to_blink:
                    blink_count += 1
                ear_consecutive_frames = 0

        # --- Overlay: kamera tam ekran, sadece yazılar üstte ---

        # Sol üst: koyu yarı saydam kutu içinde büyük beyaz sayı + "blinks"
        number_text = str(blink_count)
        number_scale = 2.2
        number_thick = 4
        (num_w, num_h), _ = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, number_scale, number_thick)
        label_text = "blinks"
        label_scale = 0.5
        label_thick = 1
        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        box_w = max(num_w, label_w) + 40
        box_h = num_h + label_h + 36
        x1, y1 = 24, 24
        x2, y2 = x1 + box_w, y1 + box_h
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (30, 32, 42), -1)
        cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 75, 95), 1)
        cv2.putText(frame, number_text, (x1 + 20, y1 + num_h + 18), cv2.FONT_HERSHEY_SIMPLEX, number_scale, (255, 255, 255), number_thick, cv2.LINE_AA)
        cv2.putText(frame, label_text, (x1 + 20, y1 + num_h + label_h + 30), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (200, 200, 210), label_thick, cv2.LINE_AA)

        # Sağ üst: "Blink Counter" başlığı
        title = "Blink Counter"
        title_scale = 0.65
        title_thick = 2
        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick)
        title_x = w - title_w - 24
        title_y = 24 + title_h + 8
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (220, 222, 235), title_thick, cv2.LINE_AA)

        cv2.imshow("Göz Kırpma Sayacı", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
