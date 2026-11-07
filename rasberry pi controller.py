import cv2
import torch
import time
import requests
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory  # Provides smooth, jitter-free PWM
import RPi.GPIO as GPIO

# ----------------- CONFIGURATION -----------------
WEIGHTS_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.75
REQUIRED_CONSECUTIVE_HITS = 4  # Must see parrot for 4 frames before springing

# Pin assignments (BCM Numbering)
SERVO_PIN = 18          # Pin 12 physically
SIGNAL_OUT_PIN = 24     # Pin 18 physically (digital trigger signal out)

# Telegram Alert Configuration
ENABLE_TELEGRAM = True
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
# -------------------------------------------------

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SIGNAL_OUT_PIN, GPIO.OUT, initial=GPIO.LOW)

# Setup Servo
# If you haven't installed pigpio, remove 'pin_factory' argument
try:
    factory = PiGPIOFactory()
    servo = Servo(SERVO_PIN, pin_factory=factory)
except Exception:
    servo = Servo(SERVO_PIN)

def reset_latch():
    """Sets servo to holding position."""
    servo.min()  # Holds the door pin
    time.sleep(0.5)
    servo.detach() # Stop sending pulses to prevent motor buzzing/heating

def release_latch():
    """Rotates servo to pull pin and drop trap door."""
    print("Actuating Servo -> Springing trap!")
    servo.max()  # Pulls the latch
    time.sleep(1.0)
    servo.detach()

def send_alert(image_path):
    if not ENABLE_TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": "🚨 Parrot captured! Trap sprung."}
        try:
            requests.post(url, data=payload, files={"photo": img}, timeout=8)
            print("Alert notification sent successfully.")
        except Exception as e:
            print(f"Failed to send alert: {e}")

# 1. Initialize Hardware
reset_latch()

# 2. Load Trained YOLOv5 Model
print("Loading model onto Raspberry Pi...")
model = torch.hub.load('yolov5', 'custom', path=WEIGHTS_PATH, source='local')
model.conf = CONFIDENCE_THRESHOLD

# 3. Start Camera (Use 0 for USB webcam, or cv2.VideoCapture(0) / Picamera2)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

consecutive_hits = 0
trap_sprung = False

print("Trap is armed and monitoring for parrots...")

try:
    while cap.isOpened() and not trap_sprung:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)
        detections = results.pandas().xyxy[0]

        target_detected = False
        for _, row in detections.iterrows():
            # Check if detection matches target class
            if row['confidence'] >= CONFIDENCE_THRESHOLD:
                target_detected = True
                print(f"Parrot detected! Confidence: {row['confidence']:.2f}")
                break

        if target_detected:
            consecutive_hits += 1
            if consecutive_hits >= REQUIRED_CONSECUTIVE_HITS:
                print("Target confirmed inside trap zone!")

                # Step A: Fire electrical signal pin HIGH
                GPIO.output(SIGNAL_OUT_PIN, GPIO.HIGH)

                # Step B: Spring the trap
                release_latch()

                # Step C: Save snapshot image
                snapshot_path = "parrot_caught.jpg"
                cv2.imwrite(snapshot_path, frame)

                # Step D: Send remote notification
                send_alert(snapshot_path)

                trap_sprung = True
        else:
            consecutive_hits = 0

finally:
    cap.release()
    GPIO.cleanup()
    print("Trap controller shutdown.")