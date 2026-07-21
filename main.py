import cv2
import requests
import mediapipe as mp
import time
import math
import threading
import queue
import json
import os
import pyttsx3
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import logging
from datetime import datetime

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('control_panel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# Configuration Manager with Validation
# ==========================================
@dataclass
class AppConfig:
    ip: str = "192.168.8.167"
    min_distance: float = 0.04
    max_distance: float = 0.20
    camera: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    voice_rate: int = 150
    hand_confidence: float = 0.4
    face_confidence: float = 0.4
    smoothing_factor: float = 0.2
    brightness_min: int = 0
    brightness_max: int = 100
    relay_count_min: int = 1
    relay_count_max: int = 8
    ping_timeout: float = 0.5
    gesture_recognition_enabled: bool = True

    def validate(self):
        """Validate configuration values"""
        assert 0 <= self.min_distance <= 1, "min_distance must be between 0 and 1"
        assert 0 <= self.max_distance <= 1, "max_distance must be between 0 and 1"
        assert self.min_distance < self.max_distance, "min_distance must be less than max_distance"
        assert 0 <= self.hand_confidence <= 1, "hand_confidence must be between 0 and 1"
        assert 0 <= self.face_confidence <= 1, "face_confidence must be between 0 and 1"
        assert 0 < self.smoothing_factor <= 1, "smoothing_factor must be between 0 and 1"


class ConfigManager:
    def __init__(self, filename="config.json"):
        self.filename = filename
        self.config = AppConfig()
        self.load()
        self.config.validate()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        else:
            self.save()
            logger.info("Default configuration created")

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.config.__dict__, f, indent=4)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")


# ==========================================
# Enhanced Voice Assistant (Fixed Threading)
# ==========================================
class VoiceAssistant(threading.Thread):
    def __init__(self, rate: int = 150):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.rate = rate
        self.engine = None
        self.is_running = True
        logger.info("Voice Assistant initialized")

    def _initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None

    def run(self):
        self._initialize_engine()
        while self.is_running:
            try:
                text = self.q.get(timeout=1)
                if text is None:
                    break
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Voice assistant error: {e}")

    def speak(self, text: str):
        if text and self.engine:
            self.q.put(text)

    def stop(self):
        self.is_running = False
        self.q.put(None)


# ==========================================
# ESP32 Controller with Fixed Retry Logic
# ==========================================
class ESP32Controller(threading.Thread):
    def __init__(self, ip: str, timeout: float = 0.5):
        super().__init__(daemon=True)
        self.ip = ip
        self.timeout = timeout
        self.q = queue.Queue()
        self.status = "Disconnected"
        self.ping = 0
        self.last_relay = -1
        self.last_brightness = -1
        self.retry_count = 0
        self.max_retries = 3
        self.session = requests.Session()
        self.session.headers.update({'Connection': 'close'})
        self.is_running = True
        logger.info(f"ESP32 Controller initialized for IP: {ip}")

    def run(self):
        while self.is_running:
            try:
                task = self.q.get(timeout=1)
                if task is None:
                    break

                cmd_type = task.get("type")
                val = task.get("value")
                retry = task.get("retry", 0)

                if cmd_type == "relay":
                    target_url = f"http://{self.ip}/set?count={val}"
                elif cmd_type == "brightness":
                    target_url = f"http://{self.ip}/brightness?value={val}"
                elif cmd_type == "retry":
                    target_url = task.get("url")
                    if target_url:
                        self._send_request(target_url, retry)
                    continue
                else:
                    continue

                self._send_request(target_url, retry)
                self.q.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ESP32 controller error: {e}")

    def _send_request(self, url: str, retry: int = 0):
        start_time = time.time()
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                self.ping = int((time.time() - start_time) * 1000)
                self.status = "Connected"
                self.retry_count = 0
            else:
                self._handle_failure(url, retry, f"HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            self._handle_failure(url, retry, str(e))

    def _handle_failure(self, url: str, retry: int, error: str = ""):
        self.status = "Disconnected"
        self.ping = 0
        self.retry_count += 1

        if self.retry_count <= self.max_retries:
            self.q.put({"type": "retry", "url": url, "retry": self.retry_count})
        else:
            logger.error(f"Max retries exceeded for {url}")

    def set_relays(self, count: int):
        if count != self.last_relay:
            self.q.put({"type": "relay", "value": count})
            self.last_relay = count

    def set_brightness(self, level: int):
        if abs(level - self.last_brightness) >= 2:
            self.q.put({"type": "brightness", "value": level})
            self.last_brightness = level

    def stop(self):
        self.is_running = False
        self.q.put(None)
        self.session.close()


# ==========================================
# Enhanced Gesture Recognition
# ==========================================
class GestureType(Enum):
    OK = "👌 OK"
    THUMB = "👍 Thumb"
    FIST = "✊ Fist"
    OPEN = "🖐 Open"
    PEACE = "✌ Peace"
    ONE = "👆 One"
    THREE = "🤟 Three"
    FOUR = "🖖 Four"
    UNKNOWN = "❓ Unknown"


class GestureRecognizer:
    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def calculate_distance(self, pt1, pt2) -> float:
        return math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

    def is_finger_extended(self, tip_idx: int, dip_idx: int, pip_idx: int, landmarks) -> bool:
        tip = landmarks[tip_idx]
        dip = landmarks[dip_idx]
        pip = landmarks[pip_idx]
        return tip.y < pip.y and tip.y < dip.y

    def _is_thumb_extended(self, landmarks) -> bool:
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]
        dist = self.calculate_distance(thumb_tip, index_mcp)
        return dist > 0.1 and thumb_tip.y < thumb_ip.y

    def recognize(self, landmarks) -> Tuple[GestureType, float]:
        thumb_extended = self._is_thumb_extended(landmarks)
        index_extended = self.is_finger_extended(8, 6, 5, landmarks)
        middle_extended = self.is_finger_extended(12, 10, 9, landmarks)
        ring_extended = self.is_finger_extended(16, 14, 13, landmarks)
        pinky_extended = self.is_finger_extended(20, 18, 17, landmarks)

        pinch_dist = self.calculate_distance(landmarks[4], landmarks[8])
        fingers = [index_extended, middle_extended, ring_extended, pinky_extended]
        extended_count = sum(fingers)

        if pinch_dist < 0.04 and not any(fingers):
            return GestureType.OK, 0.95
        if thumb_extended and not any(fingers):
            return GestureType.THUMB, 0.90
        if not thumb_extended and not any(fingers):
            return GestureType.FIST, 0.95
        if thumb_extended and extended_count == 4:
            return GestureType.OPEN, 0.95
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return GestureType.PEACE, 0.92
        if index_extended and not any(fingers[1:]):
            return GestureType.ONE, 0.90
        if index_extended and middle_extended and ring_extended and not pinky_extended:
            return GestureType.THREE, 0.85
        if all(fingers):
            return GestureType.FOUR, 0.80

        return GestureType.UNKNOWN, 0.5


# ==========================================
# Statistics Tracker
# ==========================================
class StatisticsTracker:
    def __init__(self):
        self.fps_history = []
        self.max_history = 100
        self.gesture_counts = {gesture: 0 for gesture in GestureType}
        self.total_frames = 0
        self.processing_times = []

    def update_fps(self, fps: float):
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)

    def update_gesture(self, gesture: GestureType):
        self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1

    def update_processing_time(self, time_ms: float):
        self.processing_times.append(time_ms)
        if len(self.processing_times) > self.max_history:
            self.processing_times.pop(0)

    def get_average_fps(self) -> float:
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)

    def get_average_processing_time(self) -> float:
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)


# ==========================================
# Main Application with Geometric Face Mesh
# ==========================================
class AdvancedControlPanel:
    def __init__(self):
        self.config_mgr = ConfigManager()
        self.config = self.config_mgr.config

        self.voice = VoiceAssistant(rate=self.config.voice_rate)
        self.voice.start()

        self.esp32 = ESP32Controller(self.config.ip, timeout=self.config.ping_timeout)
        self.esp32.start()

        self.cap = cv2.VideoCapture(self.config.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self.latest_hand = None
        self.latest_face = None
        self.smooth_relay_dist = 0.0
        self.smooth_bright_dist = 0.0
        self.brightness_level = 0
        self.left_gesture = GestureType.UNKNOWN
        self.right_gesture = GestureType.UNKNOWN
        self.relay_count = 1
        self.face_present = False
        self.is_recording = False
        self.video_writer = None
        self.last_voice_time = 0
        self.voice_cooldown = 1.0

        self.gesture_recognizer = GestureRecognizer()
        self.stats = StatisticsTracker()

        self.frame_skip = 1
        self.process_counter = 0

        self._check_model_files()
        self.setup_mediapipe()

        self.voice.speak("System initialized. Biometric face mesh and hand recognition active.")
        logger.info("Advanced Control Panel initialized")

    def _check_model_files(self):
        required = ["hand_landmarker.task", "face_landmarker.task"]
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            msg = f"Missing model file(s): {', '.join(missing)}. Please download them."
            logger.error(msg)
            raise FileNotFoundError(msg)

    def setup_mediapipe(self):
        try:
            hand_opts = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="hand_landmarker.task",
                    delegate=python.BaseOptions.Delegate.CPU
                ),
                num_hands=2,
                min_hand_detection_confidence=self.config.hand_confidence,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                running_mode=vision.RunningMode.IMAGE
            )
            self.hand_tracker = vision.HandLandmarker.create_from_options(hand_opts)

            face_opts = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="face_landmarker.task",
                    delegate=python.BaseOptions.Delegate.CPU
                ),
                num_faces=1,
                min_face_detection_confidence=self.config.face_confidence,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                running_mode=vision.RunningMode.IMAGE
            )
            self.face_tracker = vision.FaceLandmarker.create_from_options(face_opts)

            logger.info("MediaPipe trackers initialized successfully in IMAGE mode")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

    def calculate_distance(self, pt1, pt2) -> float:
        return math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

    def process_face(self, frame):
        with self._lock:
            face_data = self.latest_face
        self.face_present = False
        h, w = frame.shape[:2]

        if face_data and face_data.face_landmarks:
            self.face_present = True
            for face in face_data.face_landmarks:
                self._draw_geometric_face_mesh(frame, face, w, h)

    def _draw_geometric_face_mesh(self, frame, face, w, h):
        """Draws a plexus-style neural/constellation face mesh matching the reference image."""
        try:
            # 1. Draw thin, faint connecting lines (Darker Cyan/Blue)
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                    start_idx, end_idx = connection
                    if start_idx < len(face) and end_idx < len(face):
                        pt1 = (int(face[start_idx].x * w), int(face[start_idx].y * h))
                        pt2 = (int(face[end_idx].x * w), int(face[end_idx].y * h))
                        # BGR format: Faint blue-cyan
                        cv2.line(frame, pt1, pt2, (180, 100, 20), 1, cv2.LINE_AA)
            else:
                num_pts = len(face)
                for i in range(0, num_pts - 1):
                    pt1 = (int(face[i].x * w), int(face[i].y * h))
                    pt2 = (int(face[i + 1].x * w), int(face[i + 1].y * h))
                    cv2.line(frame, pt1, pt2, (180, 100, 20), 1, cv2.LINE_AA)

            # 2. Draw nodes/particles (Bright Cyan dots)
            for i, lm in enumerate(face):
                pt = (int(lm.x * w), int(lm.y * h))
                # Base small dot for all points
                cv2.circle(frame, pt, 1, (255, 220, 50), -1, cv2.LINE_AA)
                # Slightly larger, brighter dots scattered to simulate the "constellation" effect
                if i % 8 == 0:
                    cv2.circle(frame, pt, 2, (255, 255, 150), -1, cv2.LINE_AA)

            # 3. Draw intense glowing eyes
            eye_indices = [468, 473] if len(face) > 470 else [33, 263, 362, 133]
            for idx in eye_indices:
                if idx < len(face):
                    pt = (int(face[idx].x * w), int(face[idx].y * h))
                    # Multi-layered circles to create an intense additive glow illusion
                    cv2.circle(frame, pt, 16, (150, 50, 0), -1, cv2.LINE_AA)   # Faint deep blue outer aura
                    cv2.circle(frame, pt, 10, (255, 120, 0), -1, cv2.LINE_AA)  # Mid-bright blue/cyan
                    cv2.circle(frame, pt, 5, (255, 220, 100), -1, cv2.LINE_AA) # Bright cyan inner
                    cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA) # Hot white core

        except Exception as e:
            logger.error(f"Neural plexus mesh drawing error: {e}")

        # Bounding box & Status text
        x_coords = [lm.x for lm in face]
        y_coords = [lm.y for lm in face]
        xmin, ymin = int(min(x_coords) * w), int(min(y_coords) * h)
        xmin, ymin = max(0, xmin - 20), max(0, ymin - 20)

        cv2.putText(frame, "NEURAL PLEXUS ACTIVE", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 50), 2)

    def process_hands(self, frame):
        with self._lock:
            hand_data = self.latest_hand

        h, w = frame.shape[:2]
        if not hand_data or not hand_data.hand_landmarks:
            return

        for idx, hand_landmarks in enumerate(hand_data.hand_landmarks):
            raw_label = hand_data.handedness[idx][0].category_name
            # Correcting handedness for mirrored webcam view
            label = "Right" if raw_label == "Left" else "Left"

            gesture, confidence = self.gesture_recognizer.recognize(hand_landmarks)
            self.stats.update_gesture(gesture)
            self._draw_hand_landmarks(frame, hand_landmarks, label, gesture, confidence)

            if not self.face_present:
                cv2.putText(frame, "SYSTEM LOCKED: FACE NOT DETECTED",
                            (int(w / 2) - 200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                continue

            raw_dist = self.calculate_distance(hand_landmarks[4], hand_landmarks[8])

            if label == "Left":
                self.left_gesture = gesture
                self._update_relay_control(raw_dist)
            elif label == "Right":
                self.right_gesture = gesture
                self._update_brightness_control(raw_dist)

    def _draw_hand_landmarks(self, frame, landmarks, label, gesture, confidence):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
            (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
        ]

        glow_color = (255, 0, 0) if label == "Left" else (0, 150, 255)

        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], glow_color, 4)
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)

        for i, pt in enumerate(pts):
            size = 3 if i in [4, 8, 12, 16, 20] else 2
            cv2.circle(frame, pt, size + 2, (255, 255, 255), -1)
            cv2.circle(frame, pt, size, (0, 0, 255), -1)

        gesture_text = f"{label}: {gesture.value} ({confidence:.0%})"
        cv2.putText(frame, gesture_text, (pts[0][0] - 30, pts[0][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, glow_color, 2)

    def _update_relay_control(self, raw_dist):
        alpha = self.config.smoothing_factor
        self.smooth_relay_dist = (1 - alpha) * self.smooth_relay_dist + alpha * raw_dist

        min_d = self.config.min_distance
        max_d = self.config.max_distance

        if self.smooth_relay_dist < min_d:
            target_relay = self.config.relay_count_min
        elif self.smooth_relay_dist > max_d:
            target_relay = self.config.relay_count_max
        else:
            normalized = (self.smooth_relay_dist - min_d) / (max_d - min_d)
            target_relay = self.config.relay_count_min + int(
                normalized * (self.config.relay_count_max - self.config.relay_count_min)
            )

        if target_relay != self.relay_count:
            self.relay_count = target_relay
            self.esp32.set_relays(self.relay_count)

            current_time = time.time()
            if current_time - self.last_voice_time > self.voice_cooldown:
                self.voice.speak(f"Relay {self.relay_count}")
                self.last_voice_time = current_time

    def _update_brightness_control(self, raw_dist):
        alpha = self.config.smoothing_factor
        self.smooth_bright_dist = (1 - alpha) * self.smooth_bright_dist + alpha * raw_dist

        min_d = self.config.min_distance
        max_d = self.config.max_distance
        if self.smooth_bright_dist < min_d:
            bright_val = 0
        elif self.smooth_bright_dist > max_d:
            bright_val = 100
        else:
            bright_val = ((self.smooth_bright_dist - min_d) / (max_d - min_d)) * 100
        self.brightness_level = int(max(0, min(100, bright_val)))

        self.esp32.set_brightness(self.brightness_level)

    def draw_hud(self, frame, fps):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 420), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        y_pos = 30
        cv2.putText(frame, "BIOMETRIC CONTROL PANEL", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.line(frame, (20, y_pos + 10), (300, y_pos + 10), (100, 100, 100), 1)
        y_pos += 30

        cv2.putText(frame, f"FPS: {int(fps)}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        avg_processing = self.stats.get_average_processing_time()
        cv2.putText(frame, f"Process: {avg_processing:.0f}ms", (140, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 30

        wifi_color = (0, 255, 0) if self.esp32.status == "Connected" else (0, 0, 255)
        cv2.putText(frame, f"WiFi: {self.esp32.status}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, wifi_color, 2)
        cv2.putText(frame, f"Ping: {self.esp32.ping} ms", (200, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 30

        face_status = "AUTHORIZED" if self.face_present else "LOCKED"
        face_color = (0, 255, 0) if self.face_present else (0, 0, 255)
        cv2.putText(frame, f"User: {face_status}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        y_pos += 30

        cv2.putText(frame, f"L: {self.left_gesture.value}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        cv2.putText(frame, f"R: {self.right_gesture.value}", (140, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        y_pos += 40

        cv2.putText(frame, f"Relay Dist: {self.smooth_relay_dist:.3f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bar_w = 280
        relay_fill = int(max(0, min(1,
                                    (self.smooth_relay_dist - self.config.min_distance) /
                                    (self.config.max_distance - self.config.min_distance)
                                    )) * bar_w)
        cv2.rectangle(frame, (20, y_pos + 10), (20 + bar_w, y_pos + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, y_pos + 10), (20 + relay_fill, y_pos + 20), (0, 255, 0), -1)
        y_pos += 35

        cv2.putText(frame, f"Relay: {self.relay_count}/8", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for i in range(1, 9):
            color = (0, 255, 0) if i <= self.relay_count else (100, 100, 100)
            cv2.circle(frame, (20 + i * 30, y_pos + 25), 10, color, -1)
            cv2.circle(frame, (20 + i * 30, y_pos + 25), 10, (50, 50, 50), 2)
        y_pos += 55

        cv2.putText(frame, f"Brightness: {self.brightness_level}%", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        bright_fill = int((self.brightness_level / 100.0) * bar_w)
        cv2.rectangle(frame, (20, y_pos + 10), (20 + bar_w, y_pos + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, y_pos + 10), (20 + bright_fill, y_pos + 20), (0, 255, 255), -1)

        if self.is_recording:
            cv2.circle(frame, (w - 40, 40), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 90, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def run(self):
        prev_time = time.time()
        logger.info("Application started successfully")

        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    logger.warning("Failed to read frame")
                    break

                frame = cv2.flip(frame, 1)
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                self.stats.update_fps(fps)

                start_time = time.time()

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                hand_result = self.hand_tracker.detect(mp_image)
                face_result = self.face_tracker.detect(mp_image)

                with self._lock:
                    self.latest_hand = hand_result
                    self.latest_face = face_result

                self.process_face(frame)
                self.process_hands(frame)

                processing_time = (time.time() - start_time) * 1000
                self.stats.update_processing_time(processing_time)

                frame = self.draw_hud(frame, fps)

                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)

                cv2.imshow("Advanced Control Panel", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    cv2.imwrite(filename, frame)
                    self.voice.speak("Screenshot saved")
                elif key == ord('r'):
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"record_{timestamp}.avi"
                        h, w = frame.shape[:2]
                        self.video_writer = cv2.VideoWriter(
                            filename,
                            cv2.VideoWriter_fourcc(*'XVID'),
                            20.0,
                            (w, h)
                        )
                        self.voice.speak("Recording started")
                    else:
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        self.voice.speak("Recording stopped")
                elif key == ord('c'):
                    self.stats = StatisticsTracker()
                    self.voice.speak("Statistics cleared")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up resources...")
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        try:
            if hasattr(self, 'hand_tracker'):
                self.hand_tracker.close()
            if hasattr(self, 'face_tracker'):
                self.face_tracker.close()
        except Exception as e:
            logger.error(f"Error closing trackers: {e}")
        try:
            self.esp32.stop()
            self.voice.stop()
        except Exception as e:
            logger.error(f"Error stopping threads: {e}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logger.info("Starting Advanced Control Panel...")
    try:
        app = AdvancedControlPanel()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Application terminated")