import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
from gtts import gTTS

# --- 1. PREPARE AUDIO (TTS OTOMATIS) ---
def prepare_audio():
    audio_files = {
        "suara_mendekat.mp3": "Ada orang yang mendekati barang",
        "suara_ambil.mp3": "Ada orang yang ingin mengambil barang",
        "suara_dicuri.mp3": "Peringatan! Barang telah dicuri"
    }
    for filename, text in audio_files.items():
        if not os.path.exists(filename):
            tts = gTTS(text=text, lang='id')
            tts.save(filename)

prepare_audio()

# --- 2. INISIALISASI AUDIO ---
pygame.mixer.init()
try:
    sound_mendekat = pygame.mixer.Sound("suara_mendekat.mp3")
    sound_ambil    = pygame.mixer.Sound("suara_ambil.mp3")
    sound_dicuri   = pygame.mixer.Sound("suara_dicuri.mp3")
except Exception as e:
    print("Audio Error:", e)

channel_warning = pygame.mixer.Channel(1)

if os.path.exists("backsound.mp3"):
    pygame.mixer.music.load("backsound.mp3")
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play(-1)

# --- 3. MEDIAPIPE HANDS ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Koordinat 2 Blok (Sensor) - Disesuaikan agar tidak menabrak teks di atas
ZONA_PENDEKATAN = (120, 150, 520, 460)  # Blok Luar (y_min dinaikkan ke 150 agar tidak tabrakan)
TEMPAT_BARANG   = (220, 240, 420, 430)  # Blok Dalam

baseline_crop = None
status_terakhir = "AMAN"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    tangan_terdeteksi = False
    tangan_di_zona_pendekatan = False
    tangan_di_tempat_barang = False
    koordinat_jari = None

    if results.multi_hand_landmarks:
        tangan_terdeteksi = True
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing Skeleton Tangan (Merah & Putih)
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Ambil koordinat Ujung Jari Telunjuk (Index 8)
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)
            koordinat_jari = (cx, cy)

            # Gambar Lingkaran Kuning Menyala di Ujung Jari
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

            # Check Landmark per Zona
            for lm in hand_landmarks.landmark:
                lx, ly = int(lm.x * w), int(lm.y * h)
                if (ZONA_PENDEKATAN[0] < lx < ZONA_PENDEKATAN[2]) and (ZONA_PENDEKATAN[1] < ly < ZONA_PENDEKATAN[3]):
                    tangan_di_zona_pendekatan = True
                if (TEMPAT_BARANG[0] < lx < TEMPAT_BARANG[2]) and (TEMPAT_BARANG[1] < ly < TEMPAT_BARANG[3]):
                    tangan_di_tempat_barang = True

# Check Perubahan Pixel Barang
    crop_barang = frame[TEMPAT_BARANG[1]:TEMPAT_BARANG[3], TEMPAT_BARANG[0]:TEMPAT_BARANG[2]]
    gray_crop = cv2.cvtColor(crop_barang, cv2.COLOR_BGR2GRAY)
    gray_crop = cv2.GaussianBlur(gray_crop, (21, 21), 0)

    if baseline_crop is None:
        baseline_crop = gray_crop
        continue

    diff = cv2.absdiff(baseline_crop, gray_crop)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    perubahan_pixel = (np.count_nonzero(thresh) / thresh.size) * 100

    color_pendekatan = (0, 255, 255) # Kuning
    color_barang = (0, 255, 0)      # Hijau

    # --- PENENTUAN STATUS TEKS ---
    teks_status = "BARANG AMAN"
    warna_status = (0, 255, 0)

    if perubahan_pixel > 30.0 and not tangan_di_tempat_barang:
        teks_status = f"BARANG HILANG! ({perubahan_pixel:.0f}%)"
        warna_status = (0, 0, 255) # Merah
        if status_terakhir != "DICURI":
            status_terakhir = "DICURI"
            if not channel_warning.get_busy():
                channel_warning.play(sound_dicuri)

    elif tangan_di_tempat_barang:
        color_barang = (0, 0, 255)
        teks_status = "INGIN MENGAMBIL BARANG!"
        warna_status = (0, 0, 255)
        if status_terakhir != "AMBIL":
            status_terakhir = "AMBIL"
            if not channel_warning.get_busy():
                channel_warning.play(sound_ambil)

    elif tangan_di_zona_pendekatan:
        color_pendekatan = (0, 0, 255)
        teks_status = "ADA ORANG MENDEKAT!"
        warna_status = (0, 255, 255) # Kuning
        if status_terakhir != "MENDEKAT":
            status_terakhir = "MENDEKAT"
            if not channel_warning.get_busy():
                channel_warning.play(sound_mendekat)
    else:
        status_terakhir = "AMAN"

    # --- TAMPILAN TEKS TERSTRUKTUR (DISESUAIKAN PRESISI DENGAN VIDEO) ---
    # Baris 1: Status Utama (Font disesuaikan 0.65)
    cv2.putText(frame, teks_status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna_status, 2, cv2.LINE_AA)
    
    # Baris 2: Persentase Perubahan Pixel
    cv2.putText(frame, f"Perubahan: {perubahan_pixel:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Baris 3: Status Terdeteksi MediaPipe
    teks_mp = "MEDIAPIPE: TANGAN TERDETEKSI" if tangan_terdeteksi else "MEDIAPIPE: TIDAK ADA TANGAN"
    warna_mp = (0, 255, 0) if tangan_terdeteksi else (0, 0, 255)
    cv2.putText(frame, teks_mp, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna_mp, 1, cv2.LINE_AA)

    # Baris 4: Koordinat Jari
    teks_jari = f"Jari: {koordinat_jari}" if koordinat_jari else "Jari: (-,-)"
    cv2.putText(frame, teks_jari, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # --- GAMBAR RECTANGLE KOTAK/SENSOR ---
    # Sensor 1 (Luar / Zona Pendekatan)
    cv2.rectangle(frame, (ZONA_PENDEKATAN[0], ZONA_PENDEKATAN[1]), (ZONA_PENDEKATAN[2], ZONA_PENDEKATAN[3]), color_pendekatan, 2)
    cv2.putText(frame, "ZONA PENDEKATAN", (ZONA_PENDEKATAN[0], ZONA_PENDEKATAN[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_pendekatan, 1)

    # Sensor 2 (Dalam / Tempat Barang)
    cv2.rectangle(frame, (TEMPAT_BARANG[0], TEMPAT_BARANG[1]), (TEMPAT_BARANG[2], TEMPAT_BARANG[3]), color_barang, 2)
    cv2.putText(frame, "TEMPAT BARANG", (TEMPAT_BARANG[0], TEMPAT_BARANG[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_barang, 1)

    cv2.imshow('MediaPipe - Alarm 2 Kotak', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        baseline_crop = gray_crop

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()