# 🤖 Biometric ESP32 Gesture Control Panel

An AI-powered computer vision system that combines biometric face mesh detection, hand gesture recognition, text-to-speech audio feedback, and direct ESP32 smart home control over HTTP.

---

## 🌟 Key Features

* **Biometric Security Lock**: Control commands are locked unless an authorized face is actively detected on screen.
* **Neural Plexus Face Mesh**: Custom sci-fi visual overlay tracking facial landmarks in real time.
* **Dual-Hand Gestures**:
  * **Left Hand**: Adjusts relay states (1–8) based on pinch distance.
  * **Right Hand**: Controls lighting/PWM brightness (0%–100%) based on pinch distance.
  * **Gesture Classifier**: Identifies custom gestures (*OK, Thumb, Fist, Open, Peace, One, Three, Four*).
* **ESP32 Wi-Fi Integration**: Threaded non-blocking HTTP requests with automatic retry logic and latency monitoring (ping).
* **Voice Assistant**: Real-time text-to-speech (TTS) announcements for relay status and user actions.
* **Live HUD Overlay**: Displays real-time FPS, system status, network ping, gesture states, and control bars.
* **Media & Analytics**: Built-in screenshot capture, video recording, and frame processing statistics.

---

## 📋 System Architecture & Hardware Setup

### Hardware Requirements
1. **Webcam**: Standard USB or built-in webcam (720p or higher recommended).
2. **ESP32 Microcontroller**: Flashed with an HTTP server receiving relay and brightness commands.
3. **Local Network**: PC running this script and ESP32 must be connected to the same Wi-Fi network.

### Required ESP32 HTTP Endpoints
Your ESP32 web server should listen for these HTTP GET routes:
* `GET http://<ESP32_IP>/set?count=<1-8>` — Triggers relays based on count.
* `GET http://<ESP32_IP>/brightness?value=<0-100>` — Sets brightness level.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository & Set Up Virtual Environment

```bash
# Clone or create your project directory
mkdir esp32-biometric-control
cd esp32-biometric-control

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

Create a `requirements.txt` file or install directly:

```bash
pip install opencv-python requests mediapipe pyttsx3 numpy
```

### 3. Download Required MediaPipe Model Files

The MediaPipe Tasks API requires two pretrained `.task` files placed in the project root directory:

```bash
# Download Hand Landmarker model
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Download Face Landmarker model
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## 📁 Project Structure

```text
.
├── main.py                   # Main application script
├── config.json               # Auto-generated configuration file
├── hand_landmarker.task      # MediaPipe Hand Tracking Model
├── face_landmarker.task      # MediaPipe Face Tracking Model
├── control_panel.log         # Application runtime log file
└── README.md                 # Project documentation
```

---

## ⚙️ Configuration (`config.json`)

On the first run, the app generates a default `config.json` file. You can adjust your settings directly in this file:

```json
{
    "ip": "192.168.8.167",
    "min_distance": 0.04,
    "max_distance": 0.20,
    "camera": 0,
    "frame_width": 1280,
    "frame_height": 720,
    "voice_rate": 150,
    "hand_confidence": 0.4,
    "face_confidence": 0.4,
    "smoothing_factor": 0.2,
    "brightness_min": 0,
    "brightness_max": 100,
    "relay_count_min": 1,
    "relay_count_max": 8,
    "ping_timeout": 0.5,
    "gesture_recognition_enabled": true
}
```

---

## 🚀 Running the Application

Execute the Python script:

```bash
python main.py
```

### ⌨️ Keyboard Shortcuts & Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`s`** | **Take Screenshot** | Saves the current frame as `screenshot_YYYYMMDD_HHMMSS.png` |
| **`r`** | **Toggle Recording** | Starts/Stops video recording to `record_YYYYMMDD_HHMMSS.avi` |
| **`c`** | **Clear Stats** | Resets internal FPS and processing time tracking |
| **`q`** | **Quit** | Gracefully cleans up resources and exits the program |

---

## 📊 How Gesture Controls Work

| Hand | Control Variable | Action Mechanism |
| :--- | :--- | :--- |
| **Left Hand** | **Relay Counter (1–8)** | Adjust distance between Thumb tip & Index tip |
| **Right Hand** | **Brightness Level (0–100%)** | Adjust distance between Thumb tip & Index tip |

> **Note on Hand Mirroring**: Handedness is automatically swapped internally to maintain intuitive controls on mirrored webcam displays.
