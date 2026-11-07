import cv2
import time
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO

# 1. AYARLAR
ayar = {
    "kafes_id": "IZMIR-KONAK-01",
    "lokasyon_adi": "Konak - Basmane Pilot İstasyonu",
    "acik_adres": "İzmir, Konak İlçesi, Basmane Meydanı / Kültürpark Basmane Girişi Civarı Ağaçlık Alan",
    "sorumlu_kurum": "İzmir Doğa Koruma ve Milli Parklar (DKMP) Şube Müdürlüğü",
    "eposta_ayarlari": {
        "gonderici_mail": "papagandoner43@gmail.com",
        "uygulama_sifresi": "jaxgefnfyislvfvv",
        "alici_mail": "enacar435@gmail.com"
    }
}


# 2. E-POSTA GÖNDERME FONKSİYONU
def eposta_gonder(foto_yolu, tur_adi, skor, zaman_metni):
    gonderici = ayar["eposta_ayarlari"]["gonderici_mail"]
    sifre = ayar["eposta_ayarlari"]["uygulama_sifresi"]
    alici = ayar["eposta_ayarlari"]["alici_mail"]

    msg = MIMEMultipart()
    msg["From"] = f"Kafes İzleme Sistemi <{gonderici}>"
    msg["To"] = alici
    msg["Subject"] = f"🚨 [UYARI] {ayar['kafes_id']} - İstilacı Tür Tespit Edildi!"

    html_icerik = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222;">
        <h2 style="color: #c0392b;">İstilacı Tür Algılandı ve Kafes Kapatıldı</h2>
        <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
          <tr style="background-color: #f2f2f2;"><td style="padding: 8px; border: 1px solid #ddd;"><b>Kafes Kodu:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{ayar['kafes_id']}</td></tr>
          <tr><td style="padding: 8px; border: 1px solid #ddd;"><b>Lokasyon:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{ayar['lokasyon_adi']}</td></tr>
          <tr style="background-color: #f2f2f2;"><td style="padding: 8px; border: 1px solid #ddd;"><b>Açık Adres:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{ayar['acik_adres']}</td></tr>
          <tr><td style="padding: 8px; border: 1px solid #ddd;"><b>Algılanan Canlı:</b></td><td style="padding: 8px; border: 1px solid #ddd;">Yeşil Papağan ({tur_adi})</td></tr>
          <tr style="background-color: #f2f2f2;"><td style="padding: 8px; border: 1px solid #ddd;"><b>Doğruluk Skoru:</b></td><td style="padding: 8px; border: 1px solid #ddd;">%{skor}</td></tr>
          <tr><td style="padding: 8px; border: 1px solid #ddd;"><b>Zaman:</b></td><td style="padding: 8px; border: 1px solid #ddd;">{zaman_metni}</td></tr>
        </table>
        <p style="margin-top: 15px; color: #555;"><i>* Yakalama anı fotoğrafı ekte yer almaktadır.</i></p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_icerik, "html", "utf-8"))

    with open(foto_yolu, "rb") as dosya:
        resim = MIMEImage(dosya.read())
        resim.add_header("Content-Disposition", f"attachment; filename={foto_yolu}")
        msg.attach(resim)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gonderici, sifre)
            server.send_message(msg)
        print(f"\n>>> [BAŞARILI] Rapor ve yakalama fotoğrafı {alici} adresine gönderildi!")
    except Exception as e:
        print(f"\n>>> [HATA] E-posta gönderilemedi: {e}")


# 3. KAMERA VE YAPAY ZEKA BAŞLATMA
print("Kamera açılıyor... Lütfen kameranın karşısını 2 saniye boş bırakın.")
kamera = cv2.VideoCapture(0)
time.sleep(2)  # Kameranın ışığa alışması için bekleme

# İlk boş kareyi referans olarak al
_, ilk_kare = kamera.read()
gri_ilk = cv2.cvtColor(ilk_kare, cv2.COLOR_BGR2GRAY)
gri_ilk = cv2.GaussianBlur(gri_ilk, (21, 21), 0)

# Genel kuşları tanıyan hafif model (İlk açılışta otomatik iner)
model = YOLO("yolov8n.pt")

print("Sistem nöbette: Hareket algılandığında yapay zeka devreye girecek...")

while True:
    basarili, aktif_kare = kamera.read()
    if not basarili:
        break

    # Hareket Algılama Matematiksel Farkı
    gri_aktif = cv2.cvtColor(aktif_kare, cv2.COLOR_BGR2GRAY)
    gri_aktif = cv2.GaussianBlur(gri_aktif, (21, 21), 0)
    fark = cv2.absdiff(gri_ilk, gri_aktif)
    esik = cv2.threshold(fark, 25, 255, cv2.THRESH_BINARY)[1]
    hareket_miktari = cv2.countNonZero(esik)

    # Ekranda canlı görüntüyü göster (Çıkmak için klavyeden 'q'ya basabilirsin)
    cv2.imshow("Kafes Izleme Paneli", aktif_kare)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # EĞER HAREKET VARSA YAPAY ZEKAYI UYANDIR
    if hareket_miktari > 8000:
        print("-> Hareket algılandı! Yapay zeka kareyi inceliyor...")
        sonuclar = model(aktif_kare, verbose=False)

        hedef_bulundu = False
        for sonuc in sonuclar:
            for kutu in sonuc.boxes:
                sinif_id = int(kutu.cls[0])
                tur_adi = model.names[sinif_id]
                skor = round(float(kutu.conf[0]) * 100, 2)

                # Bilgisayarda test ettiğin için model 'bird' (kuş) gördüğünde tetiklensin:
                if tur_adi == "bird" and skor >= 60.0:
                    su_an = datetime.now()
                    zaman_metni = su_an.strftime("%d.%m.%Y %H:%M:%S")
                    dosya_zamani = su_an.strftime("%Y%m%d_%H%M%S")

                    foto_adi = f"yakalama_{dosya_zamani}.jpg"
                    cv2.imwrite(foto_adi, aktif_kare)

                    print(f"\n[!!! HEDEF TESPİT EDİLDİ !!!]")
                    print(f"Tür: Kuş/Papağan | Doğruluk Skoru: %{skor}")
                    print(f"Kafes Kapağı Kapatıldı! Fotoğraf kaydedildi: {foto_adi}")

                    # Mail gönder
                    eposta_gonder(foto_adi, tur_adi, skor, zaman_metni)
                    hedef_bulundu = True
                    break
            if hedef_bulundu:
                break

        if hedef_bulundu:
            print("İşlem başarıyla tamamlandı. Program durduruluyor.")
            break

        # Kuş değilse (örneğin sadece elini salladıysan) arka planı güncelle ve nöbete devam et
        time.sleep(1)
        _, ilk_kare = kamera.read()
        gri_ilk = cv2.cvtColor(ilk_kare, cv2.COLOR_BGR2GRAY)
        gri_ilk = cv2.GaussianBlur(gri_ilk, (21, 21), 0)

kamera.release()
cv2.destroyAllWindows()