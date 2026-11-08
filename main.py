import os
import glob
import cv2
import time
import torch
import smtplib
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# UI Görünüm Teması
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


# ==================== 1. MODEL VE SİSTEM AYARLARI ====================
def find_latest_weights():
    exp_files = glob.glob("yolov5/runs/train/exp*/weights/best.pt")
    if exp_files:
        return max(exp_files, key=os.path.getmtime)
    if os.path.exists("best.pt"):
        return "best.pt"
    return "yolov5/runs/train/exp8/weights/best.pt"


MODEL_PATH = find_latest_weights()

ayar = {
    "kafes_id": "IZMIR-PILOT-01",
    "lokasyon_adi": "İzmir Kültürpark & Sulak Alan Pilot İstasyonu",
    "acik_adres": "Kültürpark Basmane Girişi Ağaçlık Alan & Gölet Çevresi",
    "sorumlu_kurum": "Doğa Koruma ve Milli Parklar (DKMP) Şube Müdürlüğü",
    "eposta_ayarlari": {
        "gonderici_mail": os.getenv("GMAIL_SENDER", "your_email@gmail.com"),
        "uygulama_sifresi": os.getenv("GMAIL_APP_PASSWORD", ""),
        "varsayilan_alici": os.getenv("DEFAULT_RECIPIENT", "your_email@gmail.com")
    }
}

CLASS_MAP = {
    "Sican": {"tr": "🐀 Sıçan (Kemirgen)", "icon": "🐀"},
    "kirmizi yanakli kaplumbaga": {"tr": "🐢 Kırmızı Yanaklı Kaplumbağa", "icon": "🐢"},
    "papa-an-dataset": {"tr": "🦜 Yeşil Papağan", "icon": "🦜"}
}


# =====================================================================


class ModernTrapApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"BioTrap OS - {ayar['kafes_id']} Kontrol Paneli")
        self.geometry("1180x820")
        self.minsize(1050, 720)

        # 1. Donanım ve Model Yükleme
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Standart CPU"

        print(f"⚡ Çalışma Donanımı: {self.device.upper()} ({self.gpu_name})")
        print(f"📁 Ağırlık Dosyası: {MODEL_PATH}")

        try:
            self.model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local', device=self.device)
        except Exception:
            self.model = torch.hub.load('.', 'custom', path=MODEL_PATH, source='local', device=self.device)

        # Durum Değişkenleri
        self.cap = None
        self.is_camera_running = False
        self.consecutive_hits = 0
        self.cooldown = False

        # --- OTOMATİK BIRAKMA DEĞİŞKENLERİ ---
        self.auto_release_enabled = True
        self.release_delay_seconds = 30
        self.countdown_remaining = 0
        self.is_counting_down = False

        self._build_modern_ui()

    def _build_modern_ui(self):
        # 2 Sütunlu Grid (Sol: Kamera, Sağ: Kontrol Paneli)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= SOL PANEL: KAMERA & HUD =================
        left_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#131822")
        left_frame.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # HUD Üst Bilgi Barı
        hud_bar = ctk.CTkFrame(left_frame, fg_color="#1b2230", corner_radius=10, height=45)
        hud_bar.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")

        self.lbl_hud_title = ctk.CTkLabel(
            hud_bar, text=f"🔴 NÖBET BEKLEMEDE | {ayar['kafes_id']}",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"), text_color="#ef4444"
        )
        self.lbl_hud_title.pack(side="left", padx=15)

        lbl_device_tag = ctk.CTkLabel(
            hud_bar, text=f"⚡ {self.device.upper()} ({self.gpu_name[:16]})",
            font=ctk.CTkFont(family="Consolas", size=11), text_color="#10b981",
            fg_color="#064e3b", corner_radius=6, padx=8, pady=2
        )
        lbl_device_tag.pack(side="right", padx=12)

        # Kamera Görüntü Ekranı
        self.display_lbl = ctk.CTkLabel(left_frame,
                                        text="Kamera Akışı Başlatılmadı\n\nBaşlamak için 'Canlı Nöbeti Başlat' butonuna tıklayın.",
                                        fg_color="#0a0d14", corner_radius=12, text_color="#64748b",
                                        font=ctk.CTkFont(family="Segoe UI", size=14))
        self.display_lbl.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")

        # Alt Hızlı Eylem Barı
        action_bar = ctk.CTkFrame(left_frame, fg_color="transparent", height=50)
        action_bar.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")

        self.btn_camera = ctk.CTkButton(
            action_bar, text="▶ Canlı Nöbeti Başlat", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#059669", hover_color="#047857", height=40, corner_radius=8, command=self.toggle_camera
        )
        self.btn_camera.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_upload = ctk.CTkButton(
            action_bar, text="📁 Fotoğraf Test Et", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563eb", hover_color="#1d4ed8", height=40, corner_radius=8, command=self.upload_photo
        )
        self.btn_upload.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # ================= SAĞ PANEL: KONTROL & AYARLAR =================
        right_panel = ctk.CTkScrollableFrame(self, corner_radius=15, fg_color="#131822")
        right_panel.grid(row=0, column=1, padx=(10, 15), pady=15, sticky="nsew")

        # Kart 1: Hedef Seçimi
        card_target = ctk.CTkFrame(right_panel, fg_color="#1b2230", corner_radius=12)
        card_target.pack(fill="x", pady=6, padx=4)
        ctk.CTkLabel(card_target, text="🎯 HEDEF SEÇİMİ", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#94a3b8").pack(anchor="w", padx=12, pady=(10, 4))

        menu_options = ["🌟 Tümü (Herhangi Biri)"] + [data["tr"] for data in CLASS_MAP.values()]
        self.target_combo = ctk.CTkComboBox(card_target, values=menu_options, height=36, corner_radius=8,
                                            state="readonly")
        self.target_combo.set(menu_options[0])
        self.target_combo.pack(fill="x", padx=12, pady=(0, 8))

        # Kart 2: Çalışma Modu
        card_mode = ctk.CTkFrame(right_panel, fg_color="#1b2230", corner_radius=12)
        card_mode.pack(fill="x", pady=6, padx=4)
        ctk.CTkLabel(card_mode, text="⚙️ ÇALIŞMA MODU", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#94a3b8").pack(anchor="w", padx=12, pady=(10, 4))

        self.mode_seg = ctk.CTkSegmentedButton(card_mode, values=["Tuzak + Mail", "Sadece İzle"], height=32)
        self.mode_seg.set("Tuzak + Mail")
        self.mode_seg.pack(fill="x", padx=12, pady=(0, 10))

        # Kart 3: Hassasiyet Slider
        card_slider = ctk.CTkFrame(right_panel, fg_color="#1b2230", corner_radius=12)
        card_slider.pack(fill="x", pady=6, padx=4)
        self.lbl_slider = ctk.CTkLabel(card_slider, text="Hassasiyet Eşiği: %60",
                                       font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_slider.pack(anchor="w", padx=12, pady=(10, 2))
        self.slider = ctk.CTkSlider(card_slider, from_=30, to=95, number_of_steps=65, command=self._slider_changed)
        self.slider.set(60)
        self.slider.pack(fill="x", padx=12, pady=(2, 12))

        # Kart 4: OTOMATİK BIRAKMA ZAMANLAYICISI
        card_release = ctk.CTkFrame(right_panel, fg_color="#1b2230", corner_radius=12)
        card_release.pack(fill="x", pady=6, padx=4)
        ctk.CTkLabel(card_release, text="⏱️ OTOMATİK BIRAKMA SAYACI", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#94a3b8").pack(anchor="w", padx=12, pady=(10, 2))

        self.switch_auto_release = ctk.CTkSwitch(
            card_release, text="Otomatik Tahliyeyi Aç", font=ctk.CTkFont(size=12),
            command=self._toggle_auto_release
        )
        self.switch_auto_release.select()
        self.switch_auto_release.pack(anchor="w", padx=12, pady=4)

        self.lbl_release_slider = ctk.CTkLabel(card_release, text="Bekleme Süresi: 30 sn", font=ctk.CTkFont(size=11))
        self.lbl_release_slider.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_release = ctk.CTkSlider(card_release, from_=5, to=300, number_of_steps=59,
                                            command=self._release_slider_changed)
        self.slider_release.set(30)
        self.slider_release.pack(fill="x", padx=12, pady=(2, 8))

        self.lbl_countdown = ctk.CTkLabel(
            card_release, text="⏳ Tahliye Sayacı: Beklemede",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#f59e0b", fg_color="#3b2300", corner_radius=6, pady=4
        )
        self.lbl_countdown.pack(fill="x", padx=12, pady=(0, 10))

        # Kart 5: E-Posta Alıcısı
        card_email = ctk.CTkFrame(right_panel, fg_color="#1b2230", corner_radius=12)
        card_email.pack(fill="x", pady=6, padx=4)
        ctk.CTkLabel(card_email, text="📧 RAPOR ALICISI", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#94a3b8").pack(anchor="w", padx=12, pady=(10, 2))
        self.email_entry = ctk.CTkEntry(card_email, height=36, corner_radius=8, font=ctk.CTkFont(size=12))
        self.email_entry.insert(0, ayar["eposta_ayarlari"]["varsayilan_alici"])
        self.email_entry.pack(fill="x", padx=12, pady=(0, 10))

        # Kart 6: Telemetri & Durum Günlüğü
        card_log = ctk.CTkFrame(right_panel, fg_color="#1b2230", corner_radius=12)
        card_log.pack(fill="x", pady=6, padx=4)
        ctk.CTkLabel(card_log, text="📡 SİSTEM DURUMU", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#94a3b8").pack(anchor="w", padx=12, pady=(10, 4))
        self.lbl_event_status = ctk.CTkLabel(
            card_log, text="🟢 Sistem Beklemede\nHedef Girişi Yok",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#10b981",
            fg_color="#062e24", corner_radius=8, pady=10
        )
        self.lbl_event_status.pack(fill="x", padx=12, pady=(0, 10))

    # ---------------- Yardımcı Ayar Fonksiyonları ----------------
    def _slider_changed(self, value):
        self.lbl_slider.configure(text=f"Hassasiyet Eşiği: %{int(value)}")

    def _toggle_auto_release(self):
        self.auto_release_enabled = self.switch_auto_release.get() == 1

    def _release_slider_changed(self, value):
        self.release_delay_seconds = int(value)
        self.lbl_release_slider.configure(text=f"Bekleme Süresi: {self.release_delay_seconds} sn")

    # ---------------- Geri Sayım ve Tahliye Motoru ----------------
    def start_release_countdown(self):
        if not self.auto_release_enabled or self.is_counting_down:
            return
        self.is_counting_down = True
        self.countdown_remaining = self.release_delay_seconds
        self._tick_countdown()

    def _tick_countdown(self):
        if self.countdown_remaining > 0 and self.cooldown:
            self.lbl_countdown.configure(
                text=f"⏳ Tahliyeye Kalan: {self.countdown_remaining} sn",
                text_color="#f59e0b", fg_color="#3b2300"
            )
            self.countdown_remaining -= 1
            self.after(1000, self._tick_countdown)
        elif self.cooldown:
            # Süre doldu: Kapıyı aç ve sistemi sıfırla
            print("\n⏰ Otomatik tahliye süresi doldu! Kapı serbest bırakılıyor...")
            self.cooldown = False
            self.is_counting_down = False
            self.lbl_countdown.configure(text="✓ Hayvan Bırakıldı - Sistem Nöbette", text_color="#10b981",
                                         fg_color="#062e24")
            self.lbl_hud_title.configure(text="🟢 CANLI NÖBETTE: Kapı Açık", text_color="#10b981")
            self.lbl_event_status.configure(text="🟢 Kafes Açıldı & Sıfırlandı\nSistem Nöbette", text_color="#10b981",
                                            fg_color="#062e24")

    # ---------------- Yapay Zeka Çıkarımı ----------------
    def evaluate_detections(self, frame_bgr):
        current_threshold = self.slider.get() / 100.0
        self.model.conf = current_threshold

        results = self.model(frame_bgr)
        annotated_frame = results.render()[0]
        detections = results.pandas().xyxy[0]

        selected_ui_choice = self.target_combo.get()
        target_detected = False
        detected_species_tr = ""
        conf_score = 0.0

        for _, row in detections.iterrows():
            raw_label = row['name']
            conf = float(row['confidence']) * 100

            species_info = CLASS_MAP.get(raw_label, {"tr": raw_label, "icon": "🎯"})
            friendly_name = species_info["tr"]

            if selected_ui_choice.startswith("🌟 Tümü") or friendly_name == selected_ui_choice:
                target_detected = True
                detected_species_tr = friendly_name
                conf_score = round(conf, 1)
                break

        return annotated_frame, target_detected, detected_species_tr, conf_score

    # ---------------- Fotoğraf Yükleme ----------------
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
            messagebox.showerror("Hata", "Görsel yüklenemedi.")
            return

        annotated, detected, species_tr, conf = self.evaluate_detections(frame)
        self._display_frame(annotated)

        mode = self.mode_seg.get()

        if detected:
            self.lbl_hud_title.configure(text=f"🚨 HEDEF BULUNDU: {species_tr}", text_color="#ef4444")
            self.lbl_event_status.configure(
                text=f"🚨 YAKALANDI: {species_tr}\nGüven: %{conf}",
                text_color="#ef4444", fg_color="#450a0a"
            )
            foto_adi = f"yakalama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(foto_adi, annotated)

            if "Tuzak" in mode:
                self.cooldown = True
                self.dispatch_email(foto_adi, species_tr, conf)
                self.start_release_countdown()
        else:
            self.lbl_event_status.configure(
                text="✓ Görsel temiz.\nHedef algılanmadı.",
                text_color="#38bdf8", fg_color="#0c2d48"
            )

    # ---------------- Canlı Kamera Akışı ----------------
    def toggle_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Hata", "Kameraya erişilemedi.")
                return
            self.is_camera_running = True
            self.cooldown = False
            self.consecutive_hits = 0
            self.btn_camera.configure(text="⏹ Nöbeti Durdur", fg_color="#dc2626", hover_color="#b91c1c")
            self.lbl_hud_title.configure(text=f"🟢 CANLI NÖBETTE: {self.target_combo.get()}", text_color="#10b981")
            self.update_stream()
        else:
            self.stop_camera()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.btn_camera.configure(text="▶ Canlı Nöbeti Başlat", fg_color="#059669", hover_color="#047857")
        self.lbl_hud_title.configure(text="🔴 NÖBET DURDURULDU", text_color="#ef4444")
        self.lbl_event_status.configure(text="Kamera Kapatıldı (İnaktif)", text_color="#94a3b8", fg_color="#1b2230")

    def update_stream(self):
        if not self.is_camera_running:
            return

        ret, frame = self.cap.read()
        if ret:
            annotated, detected, species_tr, conf = self.evaluate_detections(frame)
            self._display_frame(annotated)

            if detected:
                self.consecutive_hits += 1
                if self.consecutive_hits >= 3 and not self.cooldown:
                    mode = self.mode_seg.get()
                    foto_adi = f"yakalama_canli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(foto_adi, annotated)

                    self.lbl_hud_title.configure(text=f"🚨 TUZAK TETİKLENDİ: {species_tr} (%{conf})",
                                                 text_color="#ef4444")
                    self.lbl_event_status.configure(
                        text=f"🚨 TUZAK KAPATILDI!\n{species_tr} (%{conf})",
                        text_color="#ef4444", fg_color="#450a0a"
                    )

                    if "Tuzak" in mode:
                        self.cooldown = True
                        self.dispatch_email(foto_adi, species_tr, conf)
                        self.start_release_countdown()
                    else:
                        self.cooldown = True
            else:
                self.consecutive_hits = 0

        self.after(15, self.update_stream)

    def _display_frame(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        display_w = max(self.display_lbl.winfo_width(), 640)
        display_h = max(self.display_lbl.winfo_height(), 480)

        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(display_w, display_h))
        self.display_lbl.configure(image=ctk_image, text="")

    # ---------------- E-Posta Gönderimi ----------------
    def dispatch_email(self, foto_yolu, tur_adi, skor):
        alici = self.email_entry.get().strip()
        zaman_metni = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

        thread = threading.Thread(
            target=self._send_email_thread,
            args=(foto_yolu, tur_adi, skor, zaman_metni, alici),
            daemon=True
        )
        thread.start()

    def _send_email_thread(self, foto_yolu, tur_adi, skor, zaman_metni, alici):
        gonderici = ayar["eposta_ayarlari"]["gonderici_mail"]
        sifre = ayar["eposta_ayarlari"]["uygulama_sifresi"]

        msg = MIMEMultipart()
        msg["From"] = f"BioTrap AI İstasyonu <{gonderici}>"
        msg["To"] = alici
        msg["Subject"] = f"🚨 [KRİTİK ALARM] {ayar['kafes_id']} - {tur_adi} Yakalandı!"

        html_icerik = f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #1e293b; background-color: #f8fafc; padding: 20px;">
            <div style="max-width: 560px; margin: auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden;">
              <div style="background: #dc2626; padding: 16px 20px; color: #ffffff;">
                <h2 style="margin: 0; font-size: 18px;">🚨 İstilacı Canlı Algılandı & Kafes Kapatıldı</h2>
              </div>
              <div style="padding: 20px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                  <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 10px 0; color: #64748b;">Kafes Kodu:</td><td style="font-weight: bold; text-align: right;">{ayar['kafes_id']}</td></tr>
                  <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 10px 0; color: #64748b;">Lokasyon:</td><td style="font-weight: bold; text-align: right;">{ayar['lokasyon_adi']}</td></tr>
                  <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 10px 0; color: #64748b;">Tespit Edilen Tür:</td><td style="font-weight: bold; color: #dc2626; font-size: 16px; text-align: right;">{tur_adi}</td></tr>
                  <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 10px 0; color: #64748b;">Doğruluk Skoru:</td><td style="font-weight: bold; text-align: right;">%{skor}</td></tr>
                  <tr><td style="padding: 10px 0; color: #64748b;">Zaman:</td><td style="font-weight: bold; text-align: right;">{zaman_metni}</td></tr>
                </table>
              </div>
            </div>
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
            print(f">>> [BAŞARILI] {tur_adi} alarm raporu iletildi.")
        except Exception as e:
            print(f">>> [HATA] E-posta gönderilemedi: {e}")


if __name__ == "__main__":
    app = ModernTrapApp()
    app.mainloop()