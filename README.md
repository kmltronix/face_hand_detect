#🤖 Biometric ESP32 Gesture Control Panel

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
venv\\Scripts\\activate
# On Linux/macOS:
source venv/bin/activate
