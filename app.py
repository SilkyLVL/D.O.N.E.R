import os
import glob
import cv2
import time
import torch
import numpy as np
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from flask import Flask, render_template, Response, request, jsonify
from dotenv import load_dotenv

# .env dosyasındaki değişkenleri yükle
load_dotenv()

app = Flask(__name__)
os.makedirs("static/captures", exist_ok=True)


def find_latest_weights():
    exp_files = glob.glob("yolov5/runs/train/exp*/weights/best.pt")
    if exp_files:
        return max(exp_files, key=os.path.getmtime)
    if os.path.exists("best.pt"):
        return "best.pt"
    return "yolov5/runs/train/exp8/weights/best.pt"


MODEL_PATH = find_latest_weights()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

try:
    model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local', device=DEVICE)
except Exception:
    model = torch.hub.load('.', 'custom', path=MODEL_PATH, source='local', device=DEVICE)

# Global Durum (12 Saat / 43200 sn destekli)
system_state = {
    "status": "ARMED",
    "target": "ALL",
    "confidence": 0.60,
    "mode": "trap",
    "recipient_email": os.getenv("DEFAULT_RECIPIENT", "your_email@gmail.com"),
    "last_detection": None,
    "last_score": 0,
    "fps": 0,
    "trap_closed": False,
    "auto_release_enabled": True,
    "release_delay": 3600,  # Varsayılan 1 saat (3600 sn), 43200 sn'ye (12 saat) kadar çıkabilir
    "release_time_left": 0,
    "trap_closed_timestamp": None
}

incidents_log = []

ayar = {
    "kafes_id": "IZMIR-PILOT-01",
    "lokasyon_adi": "Kültürpark Basmane & Gölet İstasyonu",
    "eposta_ayarlari": {
        "gonderici_mail": os.getenv("GMAIL_SENDER", "SENDER_GMAIL_HERE"),
        "uygulama_sifresi": os.getenv("GMAIL_APP_PASSWORD", "")
    }
}

CLASS_MAP = {
    "Sican": {"tr": "🐀 Sıçan (Kemirgen)"},
    "kirmizi yanakli kaplumbaga": {"tr": "🐢 Kırmızı Yanaklı Kaplumbağa"},
    "papa-an-dataset": {"tr": "🦜 Yeşil Papağan"}
}


def auto_release_watchdog():
    while True:
        if system_state["trap_closed"] and system_state["auto_release_enabled"] and system_state[
            "trap_closed_timestamp"]:
            elapsed = time.time() - system_state["trap_closed_timestamp"]
            remaining = max(0, int(system_state["release_delay"] - elapsed))
            system_state["release_time_left"] = remaining

            if remaining == 0:
                print("\n⏰ Süre doldu! Web sisteminde hayvan serbest bırakıldı.")
                system_state["trap_closed"] = False
                system_state["status"] = "ARMED"
                system_state["trap_closed_timestamp"] = None
                system_state["release_time_left"] = 0
        else:
            system_state["release_time_left"] = 0
        time.sleep(1)


threading.Thread(target=auto_release_watchdog, daemon=True).start()


def send_email_async(photo_path, species_name, score, timestamp):
    def _worker():
        gonderici = ayar["eposta_ayarlari"]["gonderici_mail"]
        sifre = ayar["eposta_ayarlari"]["uygulama_sifresi"]
        alici = system_state["recipient_email"]

        if not sifre:
            print(">>> [UYARI] .env dosyasında GMAIL_APP_PASSWORD tanımlanmamış, e-posta atlandı.")
            return

        msg = MIMEMultipart()
        msg["From"] = f"BioTrap Web Station <{gonderici}>"
        msg["To"] = alici
        msg["Subject"] = f"🚨 [WEB ALARM] {ayar['kafes_id']} - {species_name} Yakalandı!"

        html = f"<h3>🚨 İstilacı Canlı Yakalandı</h3><p><b>Tür:</b> {species_name}</p><p><b>Skor:</b> %{score}</p><p><b>Zaman:</b> {timestamp}</p>"
        msg.attach(MIMEText(html, "html", "utf-8"))

        if os.path.exists(photo_path):
            with open(photo_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-Disposition", f"attachment; filename={os.path.basename(photo_path)}")
                msg.attach(img)

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=12) as server:
                server.login(gonderici, sifre)
                server.send_message(msg)
            print(f">>> [WEB E-POSTA] {species_name} raporu başarıyla gönderildi: {alici}")
        except Exception as e:
            print("Mail hatası:", e)

    threading.Thread(target=_worker, daemon=True).start()


def generate_frames():
    cap = cv2.VideoCapture(0)
    consecutive_hits = 0
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        curr_time = time.time()
        system_state["fps"] = int(1.0 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        model.conf = system_state["confidence"]
        results = model(frame)
        annotated_frame = results.render()[0]
        detections = results.pandas().xyxy[0]

        target_detected = False
        detected_species = ""
        conf_score = 0.0

        for _, row in detections.iterrows():
            raw_name = row['name']
            conf = float(row['confidence']) * 100
            chosen = system_state["target"]
            if chosen == "ALL" or chosen == raw_name:
                target_detected = True
                detected_species = CLASS_MAP.get(raw_name, {"tr": raw_name})["tr"]
                conf_score = round(conf, 1)
                break

        if target_detected and not system_state["trap_closed"]:
            consecutive_hits += 1
            if consecutive_hits >= 3:
                system_state["status"] = "TRIGGERED"
                system_state["last_detection"] = detected_species
                system_state["last_score"] = conf_score

                t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                rel_path = f"static/captures/event_{t_str}.jpg"
                cv2.imwrite(rel_path, annotated_frame)

                incidents_log.insert(0, {
                    "species": detected_species,
                    "score": conf_score,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "image": rel_path
                })

                if system_state["mode"] == "trap":
                    system_state["trap_closed"] = True
                    system_state["trap_closed_timestamp"] = time.time()
                    send_email_async(rel_path, detected_species, conf_score,
                                     datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
        else:
            consecutive_hits = 0

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', config=ayar, state=system_state, classes=CLASS_MAP)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ==================== YENİ: FOTOĞRAF YÜKLEME VE TEST ENDPOINT'İ ====================
@app.route('/api/upload_test', methods=['POST'])
def upload_test():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Dosya seçilmedi!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Geçerli bir dosya bulunamadı!"}), 400

    try:
        # Gelen resmi bellekte OpenCV formatına çevir
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Görsel formatı okunamadı!"}), 400

        # YOLO ile analiz et
        model.conf = system_state["confidence"]
        results = model(frame)
        annotated_frame = results.render()[0]
        detections = results.pandas().xyxy[0]

        target_detected = False
        detected_species = ""
        conf_score = 0.0

        for _, row in detections.iterrows():
            raw_name = row['name']
            conf = float(row['confidence']) * 100
            chosen = system_state["target"]
            if chosen == "ALL" or chosen == raw_name:
                target_detected = True
                detected_species = CLASS_MAP.get(raw_name, {"tr": raw_name})["tr"]
                conf_score = round(conf, 1)
                break

        # Analiz edilmiş görseli static klasörüne kaydet
        t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = f"static/captures/upload_{t_str}.jpg"
        cv2.imwrite(rel_path, annotated_frame)

        # Eğer hedef bulunduysa ve tuzak modu açıksa işlem yap
        if target_detected:
            incidents_log.insert(0, {
                "species": detected_species,
                "score": conf_score,
                "time": datetime.now().strftime("%H:%M:%S"),
                "image": rel_path
            })

            if system_state["mode"] == "trap":
                system_state["trap_closed"] = True
                system_state["trap_closed_timestamp"] = time.time()
                system_state["last_detection"] = detected_species
                system_state["last_score"] = conf_score
                send_email_async(rel_path, detected_species, conf_score, datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

        return jsonify({
            "success": True,
            "detected": target_detected,
            "species": detected_species if target_detected else "Hedef Yok",
            "score": conf_score,
            "image_url": f"/{rel_path}"
        })

    except Exception as e:
        print("Upload test hatası:", e)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/telemetry')
def get_telemetry():
    return jsonify({
        "status": system_state["status"],
        "target": system_state["target"],
        "confidence": int(system_state["confidence"] * 100),
        "mode": system_state["mode"],
        "fps": system_state["fps"],
        "last_detection": system_state["last_detection"],
        "last_score": system_state["last_score"],
        "trap_closed": system_state["trap_closed"],
        "release_time_left": system_state["release_time_left"],
        "incidents": incidents_log[:6]
    })


@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    if "target" in data: system_state["target"] = data["target"]
    if "confidence" in data: system_state["confidence"] = float(data["confidence"]) / 100.0
    if "mode" in data: system_state["mode"] = data["mode"]
    if "email" in data: system_state["recipient_email"] = data["email"]
    if "auto_release" in data: system_state["auto_release_enabled"] = bool(data["auto_release"])
    if "release_delay" in data: system_state["release_delay"] = int(data["release_delay"])
    return jsonify({"success": True})


@app.route('/api/manual_trigger', methods=['POST'])
def manual_trigger():
    system_state["trap_closed"] = True
    system_state["trap_closed_timestamp"] = time.time()
    system_state["status"] = "MANUALLY_TRIGGERED"
    return jsonify({"success": True})


@app.route('/api/reset_trap', methods=['POST'])
def reset_trap():
    system_state["trap_closed"] = False
    system_state["status"] = "ARMED"
    system_state["trap_closed_timestamp"] = None
    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)