"""
Göz Kırpma Sayacı - Core Logic
OpenCV + MediaPipe FaceLandmarker ile gerçek zamanlı göz kırpma tespiti
face_landmarker.task modeli bu dosyayla aynı dizinde olmalıdır.
"""

import cv2
import math
import os
import time
import urllib.request
import threading
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Model ──────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# ── Göz landmark indeksleri (478-noktalı model) ────────────────────────
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# ── Parametreler ───────────────────────────────────────────────────────
EAR_THRESHOLD       = 0.22   # Bu değerin altı = göz kapalı
FRAMES_TO_BLINK     = 2      # Kaç kare üst üste kapanmalı
LOW_BLINK_THRESHOLD = 7      # Dakikada bu kadarın altı → alarm
ALARM_COOLDOWN      = 60     # Saniye


def download_model():
    """Model dosyası yoksa Storage'dan indir."""
    if not os.path.exists(MODEL_PATH):
        print("face_landmarker.task indiriliyor...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model indirildi.")


def eye_aspect_ratio(landmarks, eye_indices) -> float:
    """EAR (Eye Aspect Ratio) hesapla."""
    p1 = landmarks[eye_indices[1]]
    p2 = landmarks[eye_indices[5]]
    vertical   = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    p3 = landmarks[eye_indices[0]]
    p4 = landmarks[eye_indices[3]]
    horizontal = math.sqrt((p3.x - p4.x) ** 2 + (p3.y - p4.y) ** 2)
    return (vertical / horizontal) if horizontal else 0.0


class BlinkDetector:
    """
    Kamera döngüsünü arka planda çalıştıran göz kırpma dedektörü.

    Kullanım:
        detector = BlinkDetector()
        detector.on_blink  = lambda count: print(count)
        detector.on_frame  = lambda frame, ear: ...
        detector.on_alarm  = lambda: print("Az kırpıyorsun!")
        detector.start()
        ...
        detector.stop()

    Callbacks (hepsi opsiyonel, detector thread'inden çağrılır):
        on_blink(total_count: int)      – her kırpmada
        on_frame(bgr_frame, ear: float) – her video karesinde
        on_alarm()                      – dakikada < 7 kırpma alarmı
        on_face_lost()                  – yüz görüntüden kayboldu
        on_face_found()                 – yüz tekrar bulundu
    """

    def __init__(self):
        download_model()

        self.blink_count      = 0
        self.blink_timestamps = deque()       # Son 60 sn'deki kırpma zamanları
        self.last_alarm_time  = 0.0
        self.start_time       = time.time()

        # Dakika bazlı geçmiş (son 10 dakika)
        self.minute_history: list[int] = []
        self._minute_count  = 0
        self._minute_start  = time.time()

        self._ear_frames   = 0
        self._running      = False
        self._thread       = None
        self._face_visible = False

        # ── Callbacks ──────────────────────────────────────────────────
        self.on_blink      = None   # fn(total_count)
        self.on_frame      = None   # fn(bgr_frame, ear)
        self.on_alarm      = None   # fn()
        self.on_face_lost  = None   # fn()
        self.on_face_found = None   # fn()

    # ── Public API ─────────────────────────────────────────────────────

    def start(self):
        """Dedektörü arka plan thread'inde başlat."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Dedektörü durdur ve thread'in bitmesini bekle."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def reset(self):
        """Sayaç ve geçmişi sıfırla."""
        self.blink_count = 0
        self.blink_timestamps.clear()
        self.last_alarm_time = 0.0
        self.start_time      = time.time()
        self._minute_count   = 0
        self._minute_start   = time.time()
        self.minute_history.clear()

    def blinks_last_minute(self) -> int:
        """Son 60 saniyedeki kırpma sayısını döndür."""
        now = time.time()
        while self.blink_timestamps and now - self.blink_timestamps[0] > 60:
            self.blink_timestamps.popleft()
        return len(self.blink_timestamps)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    # ── Internal ───────────────────────────────────────────────────────

    def _loop(self):
        base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
        options   = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        cap        = cv2.VideoCapture(0)
        ts_ms      = 0

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, ts_ms)
            ts_ms += 33   # ≈30 FPS

            ear = 0.0
            if result.face_landmarks:
                if not self._face_visible:
                    self._face_visible = True
                    if self.on_face_found:
                        self.on_face_found()

                lm   = result.face_landmarks[0]
                l_ear = eye_aspect_ratio(lm, LEFT_EYE_INDICES)
                r_ear = eye_aspect_ratio(lm, RIGHT_EYE_INDICES)
                ear   = (l_ear + r_ear) / 2

                if ear < EAR_THRESHOLD:
                    self._ear_frames += 1
                else:
                    if self._ear_frames >= FRAMES_TO_BLINK:
                        self._register_blink()
                    self._ear_frames = 0
            else:
                if self._face_visible:
                    self._face_visible = False
                    if self.on_face_lost:
                        self.on_face_lost()
                self._ear_frames = 0

            # Dakika geçiş kontrolü
            now = time.time()
            if now - self._minute_start >= 60:
                self.minute_history.append(self._minute_count)
                if len(self.minute_history) > 10:
                    self.minute_history.pop(0)
                self._minute_count = 0
                self._minute_start = now

            # Alarm kontrolü
            if (now - self.start_time >= 60
                    and now - self.last_alarm_time >= ALARM_COOLDOWN
                    and self.blinks_last_minute() < LOW_BLINK_THRESHOLD):
                self.last_alarm_time = now
                if self.on_alarm:
                    self.on_alarm()

            if self.on_frame:
                self.on_frame(frame, ear)

        cap.release()
        landmarker.close()

    def _register_blink(self):
        self.blink_count     += 1
        self._minute_count   += 1
        self.blink_timestamps.append(time.time())
        if self.on_blink:
            self.on_blink(self.blink_count)