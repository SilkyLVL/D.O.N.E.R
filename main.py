import os
import cv2
import time
import torch
import smtplib
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ==================== 1. AYARLAR VE YAPILANDIRMA ====================
MODEL_PATH = "yolov5/runs/train/exp8/weights/best.pt"  # Eğittiğin YOLOv5 ağırlıkları
CONFIDENCE_THRESHOLD = 0.60                           # %60 tespit eşiği

ayar = {
    "kafes_id": "IZMIR-KONAK-01",
    "lokasyon_adi": "Konak - Basmane Pilot İstasyonu",
    "acik_adres": "İzmir, Konak İlçesi, Basmane Meydanı / Kültürpark Girişi Ağaçlık Alan",
    "sorumlu_kurum": "İzmir Doğa Koruma ve Milli Parklar (DKMP) Şube Müdürlüğü",
    "eposta_ayarlari": {
        "gonderici_mail": "papagandoner43@gmail.com",
        "uygulama_sifresi": "jaxgefnfyislvfvv",       # Arkadaşının çalışan uygulama şifresi
        "varsayilan_alici": "enacar435@gmail.com"
    }
}
# ====================================================================


class ParrotTrapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🦜 İstilacı Yeşil Papağan Tespit ve Kafes İzleme Sistemi")
        self.root.geometry("800x820")
        self.root.configure(bg="#22272e")

        # YOLOv5 Modelini Yükle
        print(f"Model yükleniyor: {MODEL_PATH}...")
        self.model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local')
        self.model.conf = CONFIDENCE_THRESHOLD

        # Durum Değişkenleri
        self.cap = None
        self.is_camera_running = False
        self.consecutive_hits = 0
        self.email_cooldown = False

        self._build_ui()

    def _build_ui(self):
        # 1. Başlık
        title_lbl = tk.Label(
            self.root, text=f"🚨 {ayar['kafes_id']} - Kafes İzleme ve Alarm Paneli",
            font=("Arial", 16, "bold"), fg="#ffffff", bg="#22272e"
        )
        title_lbl.pack(pady=10)

        # 2. Bilgi Paneli
        info_text = f"Lokasyon: {ayar['lokasyon_adi']} | Yetkili: {ayar['sorumlu_kurum']}"
        info_lbl = tk.Label(self.root, text=info_text, font=("Arial", 9), fg="#8b949e", bg="#22272e")
        info_lbl.pack(pady=2)

        # 3. Durum Çubuğu
        self.status_lbl = tk.Label(
            self.root, text="Sistem Hazır (Nöbette)",
            font=("Arial", 12, "bold"), fg="#539bf5", bg="#2d333b", width=75, pady=6
        )
        self.status_lbl.pack(pady=6)

        # 4. Alıcı E-posta Girişi
        email_frame = tk.Frame(self.root, bg="#22272e")
        email_frame.pack(pady=6)
        tk.Label(email_frame, text="Rapor Alıcı E-posta:", fg="#adbac7", bg="#22272e", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.email_entry = tk.Entry(email_frame, width=35, font=("Arial", 10))
        self.email_entry.insert(0, ayar["eposta_ayarlari"]["varsayilan_alici"])
        self.email_entry.pack(side=tk.LEFT, padx=5)

        # 5. Kontrol Butonları
        btn_frame = tk.Frame(self.root, bg="#22272e")
        btn_frame.pack(pady=6)

        self.btn_camera = tk.Button(
            btn_frame, text="▶ Kamerayı Başlat", bg="#347d39", fg="white",
            font=("Arial", 10, "bold"), width=18, command=self.toggle_camera
        )
        self.btn_camera.pack(side=tk.LEFT, padx=10)

        self.btn_upload = tk.Button(
            btn_frame, text="📁 Fotoğraf Yükle & Test Et", bg="#1f6feb", fg="white",
            font=("Arial", 10, "bold"), width=22, command=self.upload_photo
        )
        self.btn_upload.pack(side=tk.LEFT, padx=10)

        # 6. Görüntü Gösterme Alanı
        self.display_lbl = tk.Label(self.root, bg="#1c2128", width=640, height=480)
        self.display_lbl.pack(pady=10)

    # ---------------- Yapay Zeka İşleme ----------------
    def process_frame(self, frame_bgr):
        results = self.model(frame_bgr)
        annotated_frame = results.render()[0]
        detections = results.pandas().xyxy[0]

        has_parrot = len(detections) > 0
        conf = 0.0
        class_name = "Bilinmeyen"
        if has_parrot:
            conf = round(float(detections.iloc[0]['confidence']) * 100, 2)
            class_name = detections.iloc[0]['name']

        return annotated_frame, has_parrot, conf, class_name

    # ---------------- Fotoğraf Yükleme Testi ----------------
    def upload_photo(self):
        if self.is_camera_running:
            self.stop_camera()

        file_path = filedialog.askopenfilename(
            filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Hata", "Seçilen resim açılamadı.")
            return

        annotated, has_parrot, conf, class_name = self.process_frame(frame)
        self.display_image(annotated)

        if has_parrot:
            self.status_lbl.config(
                text=f"🚨 İSTİLACI PAPAĞAN TESPİT EDİLDİ (%{conf}) - KAFES KAPATILDI!",
                fg="#f85149"
            )
            zaman_metni = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            foto_adi = f"yakalama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(foto_adi, annotated)

            # E-posta gönderimini arka planda başlat
            self.dispatch_email_thread(foto_adi, class_name, conf, zaman_metni)
        else:
            self.status_lbl.config(text="✓ Bu görselde papağan algılanmadı.", fg="#539bf5")

    # ---------------- Canlı Kamera Akışı ----------------
    def toggle_camera(self):
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", "Bilgisayar kamerasına erişilemedi.")
            return

        self.is_camera_running = True
        self.email_cooldown = False
        self.consecutive_hits = 0
        self.btn_camera.config(text="⏹ Kamerayı Durdur", bg="#da3633")
        self.status_lbl.config(text="🟢 Kamera Nöbette: Canlı Görüntü İzleniyor...", fg="#56d364")
        self.update_webcam_stream()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.btn_camera.config(text="▶ Kamerayı Başlat", bg="#347d39")
        self.status_lbl.config(text="Kamera Durduruldu (Beklemede)", fg="#539bf5")

    def update_webcam_stream(self):
        if not self.is_camera_running:
            return

        ret, frame = self.cap.read()
        if ret:
            annotated, has_parrot, conf, class_name = self.process_frame(frame)
            self.display_image(annotated)

            if has_parrot:
                self.consecutive_hits += 1
                # Yanlış tetiklemeyi önlemek için art arda 3 kare tespit şartı
                if self.consecutive_hits >= 3 and not self.email_cooldown:
                    self.status_lbl.config(
                        text=f"🚨 CANLI TESPİT: PAPAĞAN YAKALANDI (%{conf}) - RAPOR GÖNDERİLİYOR!",
                        fg="#f85149"
                    )
                    zaman_metni = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                    foto_adi = f"yakalama_canli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(foto_adi, annotated)

                    self.email_cooldown = True
                    self.dispatch_email_thread(foto_adi, class_name, conf, zaman_metni)
            else:
                self.consecutive_hits = 0

        self.root.after(15, self.update_webcam_stream)

    # ---------------- Arayüzde Resim Gösterme ----------------
    def display_image(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.thumbnail((640, 480))
        img_tk = ImageTk.PhotoImage(pil_image)

        self.display_lbl.config(image=img_tk)
        self.display_lbl.image = img_tk

    # ---------------- Arkadaşının HTML E-Posta Sistemi ----------------
    def dispatch_email_thread(self, foto_yolu, tur_adi, skor, zaman_metni):
        alici = self.email_entry.get().strip()
        if not alici or "@" not in alici:
            alici = ayar["eposta_ayarlari"]["varsayilan_alici"]

        thread = threading.Thread(
            target=self._send_email_worker,
            args=(foto_yolu, tur_adi, skor, zaman_metni, alici),
            daemon=True
        )
        thread.start()

    def _send_email_worker(self, foto_yolu, tur_adi, skor, zaman_metni, alici):
        gonderici = ayar["eposta_ayarlari"]["gonderici_mail"]
        sifre = ayar["eposta_ayarlari"]["uygulama_sifresi"]

        msg = MIMEMultipart()
        msg["From"] = f"Kafes İzleme Sistemi <{gonderici}>"
        msg["To"] = alici
        msg["Subject"] = f"🚨 [UYARI] {ayar['kafes_id']} - İstilacı Tür Tespit Edildi!"

        html_icerik = f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #222;">
            <h2 style="color: #c0392b;">İstilacı Tür Algılandı ve Kafes Kapatıldı</h2>
            <table style="border-collapse: collapse; width: 100%; max-width: 520px;">
              <tr style="background-color: #f2f2f2;"><td style="padding: 8px; border: 1px solid #ddd;"><b>Kafes Kodu:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{ayar['kafes_id']}</td></tr>
              <tr><td style="padding: 8px; border: 1px solid #ddd;"><b>Lokasyon:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{ayar['lokasyon_adi']}</td></tr>
              <tr style="background-color: #f2f2f2;"><td style="padding: 8px; border: 1px solid #ddd;"><b>Açık Adres:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{ayar['acik_adres']}</td></tr>
              <tr><td style="padding: 8px; border: 1px solid #ddd;"><b>Sorumlu Kurum:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{ayar['sorumlu_kurum']}</td></tr>
              <tr style="background-color: #f2f2f2;"><td style="padding: 8px; border: 1px solid #ddd;"><b>Algılanan Canlı:</b></td><td style="padding: 8px; border: 1px solid #ddd;">Yeşil Papağan ({tur_adi})</td></tr>
              <tr><td style="padding: 8px; border: 1px solid #ddd;"><b>Doğruluk Skoru:</b></td><td style="padding: 8px; border: 1px solid #ddd;">%{skor}</td></tr>
              <tr style="background-color: #f2f2f2;"><td style="padding: 8px; border: 1px solid #ddd;"><b>Zaman:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{zaman_metni}</td></tr>
            </table>
            <p style="margin-top: 15px; color: #555;"><i>* Yakalama anı fotoğrafı ekte yer almaktadır.</i></p>
          </body>
        </html>
        """
        msg.attach(MIMEText(html_icerik, "html", "utf-8"))

        if os.path.exists(foto_yolu):
            with open(foto_yolu, "rb") as dosya:
                resim = MIMEImage(dosya.read())
                resim.add_header("Content-Disposition", f"attachment; filename={foto_yolu}")
                msg.attach(resim)

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=12) as server:
                server.login(gonderici, sifre)
                server.send_message(msg)
            print(f"\n>>> [BAŞARILI] Rapor ve yakalama fotoğrafı {alici} adresine gönderildi!")
            self.root.after(0, lambda: self.status_lbl.config(
                text=f"✓ E-posta Raporu Başarıyla Gönderildi: {alici}", fg="#56d364"
            ))
        except Exception as e:
            print(f"\n>>> [HATA] E-posta gönderilemedi: {e}")
            self.root.after(0, lambda: messagebox.showerror("E-Posta Hatası", f"Mail gönderilemedi:\n{e}"))


if __name__ == "__main__":
    root = tk.Tk()
    app = ParrotTrapApp(root)
    root.mainloop()